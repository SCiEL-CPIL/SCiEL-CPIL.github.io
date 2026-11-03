import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 1
    
    U = np.ones((n * m, rank), dtype=float)
    V = np.ones((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)
    
    return {'U': U, 'V': V, 'W': W}
