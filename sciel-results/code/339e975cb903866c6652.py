"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen multiplications for 2x2 blocks
    # M1
    U[0][col] += 1; U[3][col] += 1 # A11, A22
    V[0][col] += 1; V[4][col] += 1 # B11, B22
    W[0][col] += 1; W[3][col] += 1 # C11, C22 (Wait, W index is k*n+i)
    # C11 is i=0, k=0 -> W[0]
    # C22 is i=1, k=1 -> W[1*3+1] = W[4]
    # Let's be careful with indices.
    # U index: i*m+j
    # V index: j*p+k
    # W index: k*n+i
    
    # We need to add contributions to W for C11, C12, C21, C22.
    # C11 (i=0, k=0) -> W[0]
    # C12 (i=0, k=1) -> W[1]
    # C21 (i=1, k=0) -> W[3]
    # C22 (i=1, k=1) -> W[4]
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0, j=0 -> U[0]
    # A22: i=1, j=1 -> U[4]
    # B11: j=0, k=0 -> V[0]
    # B22: j=1, k=1 -> V[4]
    # C11: i=0, k=0 -> W[0]
    # C22: i=1, k=1 -> W[4]
    # M1 contributes to C11 and C22?
    # C11 = M1 + ...
    # C22 = M1 - ...
    # So M1 contributes +1 to W[0] and +1 to W[4].
    
    U[0][col] = 1
    U[4][col] = 1
    V[0][col] = 1
    V[4][col] = 1
    W[0][col] = 1
    W[4][col] = 1
    col += 1
    
    # M2 = (A21+A22)B11
    # A21: i=1, j=0 -> U[3]
    # A22: i=1, j=1 -> U[4]
    # B11: j=0, k=0 -> V[0]
    # C21: i=1, k=0 -> W[3]
    U[3][col] = 1
    U[4][col] = 1
    V[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # M3 = A11(B12-B22)
    # A11: i=0, j=0 -> U[0]
    # B12: j=0, k=1 -> V[1]
    # B22: j=1, k=1 -> V[4]
    # C12: i=0, k=1 -> W[1]
    U[0][col] = 1
    V[1][col] = 1
    V[4][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = A22(B21-B11)
    # A22: i=1, j=1 -> U[4]
    # B21: j=1, k=0 -> V[3]
    # B11: j=0, k=0 -> V[0]
    # C11: i=0, k=0 -> W[0]
    # C21: i=1, k=0 -> W[3]
    U[4][col] = 1
    V[3][col] = 1
    V[0][col] = -1
    W[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # M5 = (A11+A12)B22
    # A11: i=0, j=0 -> U[0]
    # A12: i=0, j=1 -> U[1]
    # B22: j=1, k=1 -> V[4]
    # C11: i=0, k=0 -> W[0]
    # C12: i=0, k=1 -> W[1]
    U[0][col] = 1
    U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1
    W[1][col] = 1
    col += 1
    
    # M6 = (A21-A11+A22)(B11+B12)
    # A21: i=1, j=0 -> U[3]
    # A11: i=0, j=0 -> U[0]
    # A22: i=1, j=1 -> U[4]
    # B11: j=0, k=0 -> V[0]
    # B12: j=0, k=1 -> V[1]
    # C22: i=1, k=1 -> W[4]
    U[3][col] = 1
    U[0][col] = -1
    U[4][col] = 1
    V[0][col] = 1
    V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A12-A22+A11)(B21+B22)
    # A12: i=0, j=1 -> U[1]
    # A22: i=1, j=1 -> U[4]
    # A11: i=0, j=0 -> U[0]
    # B21: j=1, k=0 -> V[3]
    # B22: j=1, k=1 -> V[4]
    # C11: i=0, k=0 -> W[0]
    U[1][col] = 1
    U[4][col] = -1
    U[0][col] = 1
    V[3][col] = 1
    V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Schoolbook for remaining 5 entries
    # C13 = A11B13 + A12B23 + A13B33
    # A11: U[0], B13: V[2] (j=0, k=2), C13: W[2] (i=0, k=2)
    U[0][col] = 1; V[2][col] = 1; W[2][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[2][col] = 1; col += 1 # A12: U[1], B23: V[5] (j=1, k=2)
    U[2][col] = 1; V[8][col] = 1; W[2][col] = 1; col += 1 # A13: U[2], B33: V[8] (j=2, k=2)
    
    # C23 = A21B13 + A22B23 + A23B33
    # C23: i=1, k=2 -> W[5]
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1; col += 1
    
    # C31 = A31B11 + A32B21 + A33B31
    # C31: i=2, k=0 -> W[6]
    U[6][col] = 1; V[0][col] = 1; W[6][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1
    U[8][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1
    
    # C32 = A31B12 + A32B22 + A33B32
    # C32: i=2, k=1 -> W[7]
    U[6][col] = 1; V[1][col] = 1; W[7][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[7][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[7][col] = 1; col += 1
    
    # C33 = A31B13 + A32B23 + A33B33
    # C33: i=2, k=2 -> W[8]
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    # This is for the first 3 columns of B (B1)
    # We need to do the same for the second 3 columns (B2)
    # But the indices for V and W will be different.
    # V indices for B2: j*p + (3+k)
    # W indices for B2: (3+k)*n + i
    
    # We can just repeat the logic with offset.
    
    # To simplify, we can write a helper function or just duplicate the code.
    # Given the time, I will duplicate the code for the second block.
    
    # B2 columns 3,4,5
    # V indices: 3, 4, 5 (for j=0), 6, 7, 8 (for j=1), 9, 10, 11 (for j=2)
    # W indices: 3, 4, 5 (for k=0), 6, 7, 8 (for k=1), 9, 10, 11 (for k=2)
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[3][col] = 1; V[7][col] = 1 # B11 (j=0,k=3), B22 (j=1,k=4)
    W[3][col] = 1; W[7][col] = 1 # C11 (i=0,k=3), C22 (i=1,k=4)
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[3][col] = 1
    W[6][col] = 1 # C21 (i=1,k=3)
    col += 1
    
    # M3
    U[0][col] = 1
    V[4][col] = 1; V[7][col] = -1 # B12 (j=0,k=4), B22 (j=1,k=4)
    W[4][col] = 1 # C12 (i=0,k=4)
    col += 1
    
    # M4
    U[4][col] = 1
    V[6][col] = 1; V[3][col] = -1 # B21 (j=1,k=3), B11 (j=0,k=3)
    W[3][col] = 1; W[6][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[3][col] = -1; W[4][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1; U[4][col] = 1
    V[3][col] = 1; V[4][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1; U[0][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[3][col] = 1
    col += 1
    
    # Schoolbook
    # C13 (i=0, k=5) -> W[5]
    U[0][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    U[1][col] = 1; V[8][col] = 1; W[5][col] = 1; col += 1
    U[2][col] = 1; V[11][col] = 1; W[5][col] = 1; col += 1
    
    # C23 (i=1, k=5) -> W[8]
    U[3][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[4][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    U[5][col] = 1; V[11][col] = 1; W[8][col] = 1; col += 1
    
    # C31 (i=2, k=3) -> W[9]
    U[6][col] = 1; V[3][col] = 1; W[9][col] = 1; col += 1
    U[7][col] = 1; V[6][col] = 1; W[9][col] = 1; col += 1
    U[8][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    
    # C32 (i=2, k=4) -> W[10]
    U[6][col] = 1; V[4][col] = 1; W[10][col] = 1; col += 1
    U[7][col] = 1; V[7][col] = 1; W[10][col] = 1; col += 1
    U[8][col] = 1; V[10][col] = 1; W[10][col] = 1; col += 1
    
    # C33 (i=2, k=5) -> W[11]
    U[6][col] = 1; V[5][col] = 1; W[11][col] = 1; col += 1
    U[7][col] = 1; V[8][col] = 1; W[11][col] = 1; col += 1
    U[8][col] = 1; V[11][col] = 1; W[11][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
