
import random
import random, threading, webbrowser
import socket
from flask import Flask, render_template, request, redirect, render_template
import numpy as np
import webbrowser
from fitur import *
import sounddevice as sd
import librosa
import wavio as wv
import joblib
from playsound import playsound
import os
import sounddevice as sd

file = "wav/welcome app, speech emotion, enjoy it.mp3"
os.system("mpg123 " + file)

#model

konf = ({0:'angry', 1:'disgust', 2:'fear',
        3:'happy', 4:'neutral', 5:'sad',
        6:'surprise'})


application = Flask(__name__)
random.seed()  # Initialize the random number generator
msgFromClient       = "200"
bytesToSend         = str.encode(msgFromClient)
serverAddressPort   = ("127.0.0.1", 5000)
bufferSize          = 20000
UDPClientSocket     = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)


#playsound('wav/welcome app, speech emotion, enjoy it.mp3')

def prediksi (file):
        playsound('wav/recording progress.mp3')
        freq   = 39000
        offset = 0.6
        duration  = 3
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        wv.write(file, recording, freq, sampwidth=2)
        playsound('wav/prediction wait.mp3')
        data, sampling_rate = librosa.load(file ,duration=duration, offset=offset)
        print(sampling_rate )
        sd.play(data,sampling_rate )
        sd.wait()
        d = data.reshape(1,-1)
        m = joblib.load('rf.pkl')
        pred = int(m.predict(d))
        ot = f"{ konf[pred]} { '100' } %"
        return ot


app = Flask(__name__ ,
            static_url_path='', 
            static_folder='lib',
            template_folder='template')

@app.route('/lib/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=["GET","POST"])
def upload():
    output = ""
    if request.method == "POST":
        file = request.files["file"]
        if "file" not in request.files:
            return redirect(request.url)
        if file.filename == "":
            return redirect(request.url)
        if file:
            offset=0.6
            duration=3
            # Sampling frequency
            data, sampling_rate = librosa.load(file,duration=duration, offset=offset)
            sd.play(data, sampling_rate)
            print(data.shape)
            d = data.reshape(1,-1)
            m = joblib.load('rf.pkl')
            pred = int(m.predict(d))
            print(np.array(int(pred)))        
            out = f"{ konf[pred]} { '100' } %"
        output  =out
    return render_template('upload.html',output=output)

@app.route("/realtime",methods=["GET", "POST"])
def realtime():
    ot = ""
    file = ""

    if request.method == "POST":
        if request.form.get('record') == 'record':
            
            ot = prediksi(file = "wav/recording0.wav")
    keluar = ot
    return render_template('realtime.html',metune =keluar )



if __name__ == '__main__':
    port = 8001#+ random.randint(0, 999)
    url = "http://127.0.0.1:{0}/".format(port)
    threading.Timer(1.5, lambda: webbrowser.open(url) ).start()
    app.run(host='127.0.0.1',threaded=False,port=8001)