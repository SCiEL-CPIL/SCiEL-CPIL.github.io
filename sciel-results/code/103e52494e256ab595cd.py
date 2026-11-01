import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank = n * p = 2 * 9 = 18
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            col = i * p + k
            # U: select row i of A (all j)
            for j in range(m):
                U[i * m + j][col] = 1
            # V: select column k of B (all j)
            for j in range(m):
                V[j * p + k][col] = 1
            # W: map to output position (k, i)
            W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
