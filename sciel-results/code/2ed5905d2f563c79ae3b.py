import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank is m * n
    rank = m * n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for i in range(n):
            # U index: i * m + j
            U[i * m + j][col] = 1
            # V index: j * p + k for all k
            for k in range(p):
                V[j * p + k][col] = 1
            # W index: k * n + i for all k
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
