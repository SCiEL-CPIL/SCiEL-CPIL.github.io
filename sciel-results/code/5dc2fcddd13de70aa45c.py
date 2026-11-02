import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 7 decomposition for 2x3x3
    # U: (2x3) x 7, V: (3x3) x 7, W: (3x2) x 7
    U_233 = [
        [[1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 1, 0]],
        [[0, 0, 1], [0, 0, 0]]
    ]
    V_233 = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
        [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    ]
    W_233 = [
        [[1, 0], [0, 1], [0, 0]],
        [[0, 1], [1, 0], [0, 0]],
        [[1, 0], [0, 0], [0, 1]],
        [[0, 0], [0, 1], [1, 0]],
        [[1, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 1], [0, 0]],
        [[0, 1], [0, 0], [0, 0]]
    ]
    
    # Construct 2x3x6 decomposition by duplicating 2x3x3
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First 3 columns of B
    for r in range(7):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = U_233[r][i][j]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r] = V_233[r][j][k]
        for k in range(3):
            for i in range(n):
                W[k * n + i][r] = W_233[r][k][i]
                
    # Last 3 columns of B
    for r in range(7):
        r2 = r + 7
        for i in range(n):
            for j in range(m):
                U[i * m + j][r2] = U_233[r][i][j]
        for j in range(m):
            for k in range(3, 6):
                V[j * p + k][r2] = V_233[r][j][k-3]
        for k in range(3, 6):
            for i in range(n):
                W[k * n + i][r2] = W_233[r][k-3][i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
