import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i0 in range(n):
        for k0 in range(p):
            for i in range(n):
                if i == i0:
                    for j in range(m):
                        U[i * m + j, col] = 1.0
            for j in range(m):
                V[j * p + k0, col] = 1.0
            W[k0 * n + i0, col] = 1.0
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
