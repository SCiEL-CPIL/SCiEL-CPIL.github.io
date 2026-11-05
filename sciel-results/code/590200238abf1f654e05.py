import numpy as np

def construct():
    n = 5
    rank = n
    U = np.ones((n * n, rank))
    V = np.ones((n * n, rank))
    W = np.zeros((n * n, rank))
    
    # W[k*n + i, r] = 1 if k == r, else 0
    for k in range(n):
        for i in range(n):
            idx = k * n + i
            W[idx, k] = 1.0
            
    return {'U': U, 'V': V, 'W': W}
