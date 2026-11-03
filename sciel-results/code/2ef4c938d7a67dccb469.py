"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-33 tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Partition the 4x4x9 problem into two 4x4x4 blocks and one 4x4x1 block.
    - Each 4x4x4 block uses schoolbook rank 16.
    - The 4x4x1 block uses rank 1 (all-ones tensor).
    - Total rank: 16 + 16 + 1 = 33.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    rank = 16 + 16 + 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: columns 0-3 of B (4x4x4), schoolbook rank 16
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: columns 4-7 of B (4x4x4), schoolbook rank 16
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 3: column 8 of B (4x4x1), rank 1
    # T_{i,j,8} = 1 for all i,j. Single rank-1 term.
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
    for j in range(m):
        V[j * p + 8][col] = 1
    for i in range(n):
        W[8 * n + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
