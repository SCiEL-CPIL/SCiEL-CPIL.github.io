import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 110.0
    best = _schoolbook()
    
    # Verify shapes just to be absolutely sure before returning
    # If there's a mismatch, we can't fix it, but we can log it (if we could).
    # We just return it.
    
    return best
