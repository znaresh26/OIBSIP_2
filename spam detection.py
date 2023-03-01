import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load the dataset
data = pd.read_csv('C:/Users/nares/Downloads/spam.csv', encoding='latin-1')

data

# Preprocess the text data
stopwords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(data['v2'])
y = np.array(data['v1'])

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier
classifier.fit(X_train, y_train)

# Test the classifier on the testing set
y_pred = classifier.predict(X_test)

# Calculate the accuracy score
y_pred

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
