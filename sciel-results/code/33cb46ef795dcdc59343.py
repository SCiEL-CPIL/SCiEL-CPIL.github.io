"""
Matrix multiplication 4x5x6 via schoolbook decomposition.
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6
    rank = n * m * p  # 120
    
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
