import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 120
N, M, P = 3, 3, 6

def construct():
    n, m, p = N, M, P
    
    # Rank 54 decomposition: one term for each (r, c, s)
    rank = n * p * m  # 3 * 6 * 3 = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for r in range(n):
        for c in range(p):
            for s in range(m):
                i = r * m + s
                j = s * p + c
                k = r * p + c
                U[i, idx] = 1
                V[j, idx] = 1
                W[k, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
