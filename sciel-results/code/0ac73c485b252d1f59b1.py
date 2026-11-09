import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-p solution: for each k, one term
    rank = p
    U = np.ones((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for k in range(p):
        for j in range(m):
            V[j*p+k, k] = 1
        for i in range(n):
            W[k*n+i, k] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
