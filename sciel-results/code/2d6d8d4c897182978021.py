import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 115.0

def construct():
    n, m, p = 2, 4, 8
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = 1
    
    U = np.zeros((A_dim, rank))
    V = np.zeros((B_dim, rank))
    W = np.zeros((C_dim, rank))
    
    # All-ones tensor decomposition: rank 1
    # T[i,j,l] = 1 = U[i*m+j, 0] * V[j*p+l, 0] * W[l*n+i, 0]
    # Set all entries to 1
    U[:, 0] = 1.0
    V[:, 0] = 1.0
    W[:, 0] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

