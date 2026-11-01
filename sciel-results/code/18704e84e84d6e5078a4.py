import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            # U indices: A[i,j] for i in range(n)
            for i in range(n):
                U[i * m + j][col] = 1
            # V index: B[j,k]
            V[j * p + k][col] = 1
            # W indices: C[i,k] for i in range(n)
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
