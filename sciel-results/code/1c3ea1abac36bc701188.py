import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition for 3x3x7
    # Rank is n * m * p = 3 * 3 * 7 = 63
    
    total_rank = n * m * p
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i, col j -> i*m + j
                # Index in V: row j, col k -> j*p + k
                # Index in W: row k, col i -> k*n + i (following W[k*n+i, r] convention)
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
