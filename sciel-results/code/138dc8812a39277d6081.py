import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # We decompose the multiplication into two blocks along the middle dimension m=4.
    # Block 1: A[:, 0:2] * B[0:2, :]  (3x2 * 2x7)
    # Block 2: A[:, 2:4] * B[2:4, :]  (3x2 * 2x7)
    
    # For each 3x2 * 2x7 block:
    # We treat it as 7 independent 3x2 * 2x1 multiplications (one for each column k of B).
    # The rank of 3x2 * 2x1 is 3.
    # So each block has rank 7 * 3 = 21.
    # Total rank = 21 + 21 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 12 x 42
    V = [[0] * rank for _ in range(m * p)]  # 28 x 42
    W = [[0] * rank for _ in range(p * n)]  # 21 x 42
    
    # Block 1: j in {0, 1}
    for k in range(p):
        for i in range(n):
            col = k * 3 + i
            # U indices: i*m + j. Here j is 0 or 1.
            U[i * m + 0][col] = 1
            U[i * m + 1][col] = 1
            
            # V indices: j*p + k. Here j is 0 or 1.
            V[0 * p + k][col] = 1
            V[1 * p + k][col] = 1
            
            # W indices: k*n + i
            W[k * n + i][col] = 1
            
    # Block 2: j in {2, 3}
    for k in range(p):
        for i in range(n):
            col = 21 + k * 3 + i
            # U indices: i*m + j. Here j is 2 or 3.
            U[i * m + 2][col] = 1
            U[i * m + 3][col] = 1
            
            # V indices: j*p + k. Here j is 2 or 3.
            V[2 * p + k][col] = 1
            V[3 * p + k][col] = 1
            
            # W indices: k*n + i
            W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
