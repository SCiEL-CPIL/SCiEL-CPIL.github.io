import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i*m + j, r]
    # For i=0: U[0..2, 0] = 1, U[0..2, 1] = 0
    # For i=1: U[3..5, 0] = 0, U[3..5, 1] = 1
    U[0, 0] = 1
    U[1, 0] = 1
    U[2, 0] = 1
    U[3, 1] = 1
    U[4, 1] = 1
    U[5, 1] = 1
    
    # V[j*p + k, r]
    # V is all 1s
    V[:] = 1
    
    # W[k*n + i, r]
    # For i=0 (even indices in W col 0? No, index is k*n+i)
    # k=0..9, i=0 -> idx 0, 2, 4... W[idx, 0] = 1, W[idx, 1] = 0
    # k=0..9, i=1 -> idx 1, 3, 5... W[idx, 0] = 0, W[idx, 1] = 1
    
    for k in range(p):
        W[k * n + 0, 0] = 1
        W[k * n + 0, 1] = 0
        W[k * n + 1, 0] = 0
        W[k * n + 1, 1] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
