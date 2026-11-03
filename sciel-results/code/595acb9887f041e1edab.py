"""Program for matrix multiplication 3,4,7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _build_schoolbook():
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

_cached_result = None

def construct():
    global _cached_result
    if _cached_result is None:
        _cached_result = _build_schoolbook()
    return _cached_result
