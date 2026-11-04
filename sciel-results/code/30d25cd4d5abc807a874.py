"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Rank-16 decomposition for 4x4x4 + rank-1 for 4x4x2 = rank 17
    rank_4x4x4 = 16
    rank_4x4x2 = 1
    total_rank = rank_4x4x4 + rank_4x4x2  # 17
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 17
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 17
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 17
    
    # 4x4x4 part (r = 0..15)
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U[i * m + j][r] = 1
            for k in range(4):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    # 4x4x2 part (r = 16)
    r = 16
    for i in range(4):
        for j in range(4):
            U[i * m + j][r] = 1
    for j in range(4):
        for k in range(4, 6):
            V[j * p + k][r] = 1
    for i in range(4):
        for k in range(4, 6):
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
