import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS
    total_naive = n * m * p  # 96

    # Direct naive decomposition: rank = n*m*p = 96
    # Each (i,j,k) triple gets one column
    U = [[0] * total_naive for _ in range(n * m)]
    V = [[0] * total_naive for _ in range(m * p)]
    W = [[0] * total_naive for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
