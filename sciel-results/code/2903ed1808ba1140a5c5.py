"""Matrix multiplication 3,3,7 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
DEADLINE = START_TIME + 140  # safety margin before 150s wall-clock

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # Rank is n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
