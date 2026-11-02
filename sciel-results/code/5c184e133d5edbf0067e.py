import time
import sys

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # 1. 3x3x3 schoolbook (27)
    # 2. 3x1x3 outer product (3)
    # 3. 3x3x1 outer product (3)
    # 4. 3x1x1 outer product (3)
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: i in 0..2, j=3, k in 0..2
    for i in range(3):
        for k in range(3):
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 3: i in 0..2, j in 0..2, k=3
    for i in range(3):
        for j in range(3):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
            
    # Part 4: i in 0..2, j=3, k=3
    for i in range(3):
        U[i * m + 3][col] = 1
        V[3 * p + 3][col] = 1
        W[3 * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
