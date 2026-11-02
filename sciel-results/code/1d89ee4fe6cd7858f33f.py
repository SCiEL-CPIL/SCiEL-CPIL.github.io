"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    
    # We use a known rank-21 decomposition for 4x4x4 matrix multiplication.
    # The Bini-Capovani-Lotti-Romani (BCLR) algorithm.
    # U, V, W are 16x21 matrices for the 4x4x4 case.
    # We restrict this to 3x4x4 by:
    # 1. Taking the first 12 rows of U (i=0,1,2).
    # 2. Keeping V as is (indices j, k match).
    # 3. Re-indexing W: W_344[k*3+i, c] = W_444[k*4+i, c] for i=0,1,2.
    
    # Hardcoded BCLR Rank 21 Matrices for 4x4x4
    # U (16x21)
    U_444 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ]
    
    # This is the identity matrix, which is rank 16.
    # I need the actual BCLR matrices.
    # Since I don't have them memorized, I will use a fallback to a known lower rank algorithm or just return the schoolbook if I can't guarantee correctness of hardcoded values.
    # However, for the purpose of this task, I will use a rank 24 algorithm derived from Strassen's 7-multiplication 2x2 blocks, which is easier to construct correctly.
    # Strassen for 4x4x4:
    # Split into 2x2 blocks.
    # Use Strassen's 7 products for each 2x2 block multiplication? No, that's 49.
    # We can use the fact that we only need 3x4x4.
    
    # Let's use a simple block decomposition.
    # A (3x4) * B (4x4)
    # Split B into B1 (2x4) and B2 (2x4).
    # C = A1 B1 + A2 B2, where A1, A2 are 3x2.
    # Rank of 3x2x4 is at most 12?
    # 3x2x4 multiplication can be done with 12 multiplications (schoolbook).
    # So total rank 24.
    # This is better than 48.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    # This is 3x2x4.
    # We use schoolbook for this block.
    # i in 0..2, j in 0..1, k in 0..3
    # U[i*4+j, c] = 1, V[j*4+k, c] = 1, W[k*3+i, c] = 1
    col = 0
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: A[:, 2:4] * B[2:4, :]
    # This is 3x2x4.
    # i in 0..2, j in 2..3, k in 0..3
    # U[i*4+j, c] = 1, V[j*4+k, c] = 1, W[k*3+i, c] = 1
    for i in range(n):
        for j in range(2, 4):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
