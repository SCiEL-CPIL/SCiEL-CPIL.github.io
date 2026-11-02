import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110

    U = np.zeros((n * m, rank_schoolbook), dtype=int)
    V = np.zeros((m * p, rank_schoolbook), dtype=int)
    W = np.zeros((p * n, rank_schoolbook), dtype=int)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
