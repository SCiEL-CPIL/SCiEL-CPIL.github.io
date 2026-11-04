import numpy as np

def construct():
    n, m, p = 3, 5, 5
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
