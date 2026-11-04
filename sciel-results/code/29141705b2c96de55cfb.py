import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # Trivial rank-32 decomposition for 4x4x8 matrix multiplication
    # For each (i, k) pair, we create one rank-1 term
    # r = i * p + k
    # U[i*m + j, r] = 1 for all j
    # V[j*p + k, r] = 1 for all j
    # W[k*n + i, r] = 1
    
    rank = n * p  # 4 * 8 = 32
    
    U = np.zeros((n * m, rank))  # (16, 32)
    V = np.zeros((m * p, rank))  # (32, 32)
    W = np.zeros((p * n, rank))  # (32, 32)
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            
            # U[i*m + j, r] = 1 for all j in 0..m-1
            for j in range(m):
                U[i * m + j, r] = 1.0
            
            # V[j*p + k, r] = 1 for all j in 0..m-1
            for j in range(m):
                V[j * p + k, r] = 1.0
            
            # W[k*n + i, r] = 1
            W[k * n + i, r] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
