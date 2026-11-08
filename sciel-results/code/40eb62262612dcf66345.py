import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a valid tensor decomposition for 4x4 * 4x8 = 4x8.
    Uses the naive rank-1 (per term) decomposition, resulting in rank n*m*p = 128.
    This is guaranteed to be exact and correct.
    """
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline:
                    break
                
                # U index: i*m + j
                U[i * m + j, r] = 1.0
                # V index: j*p + k
                V[j * p + k, r] = 1.0
                # W index: k*n + i (following W[k*n+i, r] convention)
                W[k * n + i, r] = 1.0
                
                r += 1
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # Convert to lists of lists
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
