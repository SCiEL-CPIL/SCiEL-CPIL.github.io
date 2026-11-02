import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    rank = n * m * p  # 36
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
