import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # Schoolbook rank is n*m*p = 32
    rank = n * m * p
    
    U = np.zeros((n * m, rank))  # (8, 32)
    V = np.zeros((m * p, rank))  # (16, 32)
    W = np.zeros((p * n, rank))  # (8, 32)
    
    # Index mapping:
    # A is n x m. Index u = i * m + j.
    # B is m x p. Index v = j * p + k.
    # C is n x p. Index w = k * n + i. (Per prompt: W[k*n+i,r])
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
