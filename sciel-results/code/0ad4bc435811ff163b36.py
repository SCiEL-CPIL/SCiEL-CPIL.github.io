import numpy as np

def construct():
    n, m, p = 3, 3, 7
    rank = 21
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 21
    V = [[0.0] * rank for _ in range(m * p)]  # 21 x 21
    W = [[0.0] * rank for _ in range(p * n)]  # 21 x 21
    
    col = 0
    # Block 1: columns 0,1,2 (3x3 * 3x3, naive rank 9)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Block 2: columns 3,4,5 (3x3 * 3x3, naive rank 9)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Block 3: column 6 (3x3 * 3x1, naive rank 3)
    for i in range(n):
        for j in range(m):
            k = 6
            U[i * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
