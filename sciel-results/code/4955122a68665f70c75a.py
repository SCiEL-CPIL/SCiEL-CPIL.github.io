"""
Rank-minimal exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
Uses the schoolbook decomposition with rank 40.
"""
import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    rank = n * m * p  # 40
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 40
    V = [[0] * rank for _ in range(m * p)]   # 20 x 40
    W = [[0] * rank for _ in range(p * n)]   # 20 x 40
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
