from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

#create two categories related to books and clothing.
corpus = [
    'i love the book',
    'this book was not so great',
    'the fit was great',
    'i love the shoes'
]

books = 'Books'
clothing = 'Clothing'

categories = [books, books, clothing, clothing] #this will be our y

# vectorizer = CountVectorizer()
vectorizer = CountVectorizer(ngram_range=(1, 2)) # our X
# vectorizer = CountVectorizer(ngram_range=(1, 4))

vectors = vectorizer.fit_transform(corpus) # get words and vectors

print(vectorizer.get_feature_names_out())

# show true / false for words that defines as important.
print(vectors.toarray())

# classification vs regression problems - reminder to study
# SVC works with vectors and classification
clf = SVC(kernel='linear')
clf.fit(vectors, categories)

test_corpus = [
    'i love this read',
    'such a nice hat',
    'what a great book',
]

test_categories = [books, clothing, books]
test_x = vectorizer.transform(test_corpus) # fit_transform vs transform - transfor get only the vectors

print(clf.predict(test_x))
print(clf.score(test_x, test_categories)) #print %