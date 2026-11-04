import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10  # Safety margin, schoolbook is instant
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Build the schoolbook decomposition
    U = np.zeros((n * m, rank_school), dtype=np.int32)
    V = np.zeros((m * p, rank_school), dtype=np.int32)
    W = np.zeros((p * n, rank_school), dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
