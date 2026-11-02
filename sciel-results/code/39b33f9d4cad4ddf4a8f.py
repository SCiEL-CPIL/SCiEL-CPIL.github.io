import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    # Rank 32 schoolbook decomposition
    # One term for each (i, j, k)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, r] = 1.0
                # V index: j*p + k
                V[j * p + k, r] = 1.0
                # W index: k*n + i
                W[k * n + i, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
