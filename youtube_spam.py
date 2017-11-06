# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 22:49:35 2017

@author: Sri Hari Rao Yerra
"""

          
import os
import pandas as pd

def calc_performance(cm):
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tp = cm[1,1]
    total = tn + fp + fn + tp
    accuracy = round((tn + tp)*100/float(total),2)
    precision = round((tp*100)/float(fp + tp),2)
    recall  = round((tp*100)/(fn + tp),2)
    f1Score  = round(2*precision*recall/float(precision+recall),2)    
    return (accuracy,precision,recall,f1Score)


os.chdir('D:\Dataset\UCI\YouTube-Spam-Collection-v1')

# Importing the Data sets
dataset = pd.read_csv('Youtube01-Psy.csv')
dataset1 = pd.read_csv('Youtube02-KatyPerry.csv')
dataset2 = pd.read_csv('Youtube03-LMFAO.csv')
dataset3 = pd.read_csv('Youtube04-Eminem.csv')
dataset4 = pd.read_csv('Youtube05-Shakira.csv')

dataset = dataset.append(dataset1,ignore_index=True)
dataset = dataset.append(dataset2,ignore_index=True)
dataset = dataset.append(dataset3,ignore_index=True)
dataset = dataset.append(dataset4,ignore_index=True)




import re

#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,len(dataset['CONTENT'])):
    ## Step1: Keep non character 
    review = re.sub('[^a-zA-Z]', ' ', dataset['CONTENT'][i])
    
    # Step2: Converting to Lowercase
    review = review.lower()
    
    #Step3: Remove stopwords
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #Step4: Stemming
    ps = PorterStemmer()
    review =[ps.stem(word) for word in review]
    
    #Step5:  Join the word to make sentence
    review = ' '.join(review)
    corpus.append(review)

#Creating the Bag of Words Model
from sklearn.feature_extraction.text import  CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


## Model : Naive Bayes with MultinomialNB
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
naive_bayes_cm = confusion_matrix(y_test, y_pred)
nb_performance = calc_performance(naive_bayes_cm)
        
# Applying KFold Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train , cv = 10)
accuracies.mean()
accuracies.std()   

## Model Performance 
data = [nb_performance]
index = ['naive bayes']
name = ['acc','prec','rec','f1']
df = pd.DataFrame(data,columns=name,index=index)