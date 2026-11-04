import numpy as np
import time

def construct():
    deadline = time.time() + 140.0
    n, m, p = 3, 4, 7
    total = n * m * p
    
    # Build the trivial rank-n*m*p decomposition using structured indexing
    U = np.zeros((n * m, total))
    V = np.zeros((m * p, total))
    W = np.zeros((p * n, total))
    
    # Map each (i,j,k) triple to a unique rank column
    for idx in range(total):
        k = idx % p
        rem = idx // p
        j = rem % m
        i = rem // m
        U[i * m + j, idx] = 1.0
        V[j * p + k, idx] = 1.0
        W[k * n + i, idx] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
