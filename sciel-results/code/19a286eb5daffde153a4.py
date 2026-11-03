"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]

# Strassen's 2x2x2 rank-7 decomposition factors
# A = A11 A12; B = B11 B12; C = C11 C12 (each 2x2 block)
# M1 = (A11+A22)*(B11+B22)
# M2 = (A11+A22)*B21
# M3 = A12*(B21-B22)
# M4 = A11*B11
# M5 = A22*(B12-B11)
# M6 = (A12+A21)*B11
# M7 = (A21-A11)*(B12+B21)
# C11 = M1+M4-M5+M7
# C12 = M3+M4+M6
# C21 = M2+M4+M5
# C22 = M1-M2+M3+M6

STRASSEN_U = [
    [1, 0, 0, 1, 0, 0, 0],  # A11
    [0, 1, 1, 0, 1, 0, 1],  # A12
    [1, 0, 0, 0, 0, 1, -1], # A21
    [0, 1, 0, 0, 1, 1, 0],  # A22
]

STRASSEN_V = [
    [1, 1, 0, 1, 0, 1, 1],  # B11
    [0, 0, 1, 0, -1, 0, 0], # B12
    [1, 1, -1, 0, 0, 0, 0], # B21
    [1, 0, -1, 0, 0, 1, 1], # B22
]

STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],  # C11
    [0, 0, 1, 1, 0, 1, 0],   # C12
    [0, 1, 0, 1, 1, 0, 0],   # C21
    [1, -1, 1, 0, 0, 1, 0],  # C22
]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Build a decomposition using Strassen for the 4x4 middle part
    # Split rows of A (3x4) into 2+1, columns of A into 2+2
    # Split rows of B (4x5) into 2+2, columns of B (5) as is
    
    # A is 3x4: split into blocks
    # A11: 2x2 (rows 0-1, cols 0-1)
    # A12: 2x2 (rows 0-1, cols 2-3)
    # A21: 1x2 (row 2, cols 0-1)
    # A22: 1x2 (row 2, cols 2-3)
    
    # B is 4x5: split into blocks
    # B11: 2x5 (rows 0-1, cols 0-4)
    # B12: 2x5 (rows 2-3, cols 0-4)
    # B21: 2x5 (rows 0-1, cols 0-4) -- wait, let me reconsider
    
    # Actually, let's use a different approach.
    # Use schoolbook for the non-square parts and Strassen for the square part.
    
    # Split the 4x4 middle dimension:
    # A: 3x4 -> rows: 2+1, cols: 2+2
    # B: 4x5 -> rows: 2+2, cols: 5
    
    # Let's think of this as:
    # C = A * B where A is 3x4, B is 4x5
    # Split A into [A_top (2x4); A_bot (1x4)]
    # Split B into [B_left (4x5)]
    # Split A's columns into A11 (2x2), A12 (2x2), A21 (1x2), A22 (1x2)
    # Split B's rows into B_top (2x5), B_bot (2x5)
    
    # This gives:
    # C_top (2x5) = A11*B_top + A12*B_bot
    # C_bot (1x5) = A21*B_top + A22*B_bot
    
    # We can use Strassen for the 2x2 matrix multiplications:
    # A11*B_top: this is (2x2) * (2x5) -> not square
    # Hmm, Strassen requires square matrices.
    
    # Alternative: Pad to make square, or use a different decomposition.
    
    # Let's just use the schoolbook decomposition which is guaranteed to work.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
