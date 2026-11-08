import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 144
    
    U = np.zeros((n*m, rank_sb), dtype=float)
    V = np.zeros((m*p, rank_sb), dtype=float)
    W = np.zeros((p*n, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
