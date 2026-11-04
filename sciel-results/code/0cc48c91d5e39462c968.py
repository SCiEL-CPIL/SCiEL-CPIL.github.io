import numpy as np

def construct():
    n, m, p = 4, 5, 6
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
        
    return {'U': U, 'V': V, 'W': W}
