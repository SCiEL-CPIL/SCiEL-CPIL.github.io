"""Matrix multiplication 2x3 x 3x8 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]
TIME_LIMIT = 150.0
START_TIME = time.time()

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME) - 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = n * m * p  # 48
    
    # Schoolbook decomposition
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
