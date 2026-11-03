import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    # 27 for 3x3x3 schoolbook
    # 9 for 3x3x1
    # 3 for 3x1x3
    # 3 for 3x1x1
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Schoolbook 3x3x3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A index: i*4 + j
                # B index: j*4 + k
                # C index: k*3 + i
                U[i * 4 + j][col] = 1
                V[j * 4 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # col is now 27
    
    # Part 2: A[0:3, 0:3] * B[0:3, 3] -> C[0:3, 3]
    # A[i, j] * B[j, 3] -> C[i, 3]
    for i in range(3):
        for j in range(3):
            U[i * 4 + j][col] = 1
            V[j * 4 + 3][col] = 1
            W[3 * 3 + i][col] = 1
            col += 1
    # col is now 36
    
    # Part 3: A[0:3, 3] * B[3, 0:3] -> C[0:3, 0:3]
    # A[i, 3] * B[3, k] -> C[i, k]
    for i in range(3):
        for k in range(3):
            U[i * 4 + 3][col] = 1
            V[3 * 4 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1
    # col is now 39
    
    # Part 4: A[0:3, 3] * B[3, 3] -> C[0:3, 3]
    # A[i, 3] * B[3, 3] -> C[i, 3]
    for i in range(3):
        U[i * 4 + 3][col] = 1
        V[3 * 4 + 3][col] = 1
        W[3 * 3 + i][col] = 1
        col += 1
    # col is now 42
    
    return {"U": U, "V": V, "W": W}
