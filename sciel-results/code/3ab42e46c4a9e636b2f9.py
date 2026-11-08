import numpy as np
import time

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 150

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - START_TIME > TIME_LIMIT - 10:
                    return {"U": U, "V": V, "W": W}
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
