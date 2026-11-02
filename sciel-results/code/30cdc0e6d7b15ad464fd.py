import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        # Map r to (i, j, k)
        # i in [0, n), j in [0, m), k in [0, p)
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        # U index: i * m + j
        U[i * m + j, r] = 1.0
        
        # V index: j * p + k
        V[j * p + k, r] = 1.0
        
        # W index: k * n + i (trace convention)
        W[k * n + i, r] = 1.0
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
