import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # Block 1: Columns 0-2 (3 cols) -> Schoolbook 3x3x3 (Rank 27)
    # Block 2: Columns 3-5 (3 cols) -> Schoolbook 3x3x3 (Rank 27)
    # Block 3: Columns 6-7 (2 cols) -> Schoolbook 3x3x2 (Rank 18)
    
    total_rank = 27 + 27 + 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    column = 0
    # Block 1: i,j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Block 2: i,j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Block 3: i,j in 0..2, k in 6..7
    for i in range(n):
        for j in range(m):
            for k in range(6, 8):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
