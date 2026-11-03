import numpy as np

def construct():
    n, m, p = 2, 4, 6
    # Decompose into two blocks of 2x2x3
    # Block 1: j in 0,1
    # Block 2: j in 2,3
    
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Block 1
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Block 2
    for i in range(n):
        for j in range(2, m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
