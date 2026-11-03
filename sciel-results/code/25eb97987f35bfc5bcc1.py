import numpy as np
import time

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 120.0

def _check_time():
    global _START_TIME
    return time.time() - _START_TIME < _DEADLINE - 5.0

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
