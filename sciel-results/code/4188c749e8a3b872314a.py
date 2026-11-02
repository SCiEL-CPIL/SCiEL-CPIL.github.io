import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    rank = n * m  # 10
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1.0
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
