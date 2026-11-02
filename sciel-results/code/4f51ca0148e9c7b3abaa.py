import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    rank = 2 * p + 2
    total_U = n * m
    total_V = m * p
    total_W = p * n
    
    U = np.zeros((total_U, rank))
    V = np.zeros((total_V, rank))
    W = np.zeros((total_W, rank))
    
    # Indices:
    # U: i * m + j
    # V: j * p + k
    # W: k * n + i
    
    # Diagonal terms: A[i,i] * B[i,k] -> C[i,k]
    # i=0: A[0,0] (idx 0), B[0,k] (idx k), C[0,k] (idx 2k)
    # i=1: A[1,1] (idx 3), B[1,k] (idx 15+k), C[1,k] (idx 2k+1)
    
    for k in range(p):
        # Term 2k: A[0,0] * B[0,k] -> C[0,k]
        r = 2 * k
        U[0, r] = 1
        V[k, r] = 1
        W[2 * k, r] = 1
        
        # Term 2k+1: A[1,1] * B[1,k] -> C[1,k]
        r = 2 * k + 1
        U[3, r] = 1
        V[15 + k, r] = 1
        W[2 * k + 1, r] = 1
        
    # Cross terms:
    # A[0,1] * B[1,k] -> C[0,k] for all k
    # A[1,0] * B[0,k] -> C[1,k] for all k
    
    # Term 2p: A[0,1] (idx 1) * B[1,k] (idx 15+k) -> C[0,k] (idx 2k)
    r = 2 * p
    U[1, r] = 1
    for k in range(p):
        V[15 + k, r] = 1
        W[2 * k, r] = 1
        
    # Term 2p+1: A[1,0] (idx 2) * B[0,k] (idx k) -> C[1,k] (idx 2k+1)
    r = 2 * p + 1
    U[2, r] = 1
    for k in range(p):
        V[k, r] = 1
        W[2 * k + 1, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
