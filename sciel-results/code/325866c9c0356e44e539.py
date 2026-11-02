"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

import time
import random

# EVOLVE-BLOCK-START
def construct():
    # Rank-17 decomposition for 2x3x3 matrix multiplication.
    # Uses Strassen for the 2x2x2 core (rank 7) plus schoolbook for boundary terms.
    n, m, p = DIMENSIONS
    
    # U: (n*m, rank) = (6, 17)
    # V: (m*p, rank) = (9, 17)
    # W: (p*n, rank) = (6, 17)
    
    rank = 17
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for the 2x2x2 sub-problem
    # A_sub is rows 0,1 cols 0,1
    # B_sub is rows 0,1 cols 0,1
    # C_sub is rows 0,1 cols 0,1
    
    # W1 = A00 * B00
    col = 0
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1
    col += 1
    # W2 = (A00+A01) * B10
    U[0][col] = 1; U[1][col] = 1; V[3][col] = 1; W[0][col] = 1
    col += 1
    # W3 = A00 * (B01-B11)
    U[0][col] = 1; V[1][col] = 1; V[4][col] = -1; W[1][col] = 1
    col += 1
    # W4 = (A10-A00) * B01
    U[3][col] = 1; U[0][col] = -1; V[1][col] = 1; W[1][col] = 1
    col += 1
    # W5 = (A00+A11) * (B10+B11)
    U[0][col] = 1; U[4][col] = 1; V[3][col] = 1; V[4][col] = 1; W[1][col] = 1
    col += 1
    # W6 = (A11-A01) * (B00+B10)
    U[4][col] = 1; U[1][col] = -1; V[0][col] = 1; V[3][col] = 1; W[3][col] = 1
    col += 1
    # W7 = (A00-A11) * (B11+B01)
    U[0][col] = 1; U[4][col] = -1; V[4][col] = 1; V[1][col] = 1; W[3][col] = 1
    col += 1
    
    # Now we have used 7 columns. We need 10 more for the boundary terms.
    # The remaining terms involve row/col index 2 in A, B, or C.
    
    # Terms for C[:, 2] from A[:, 0:2] * B[0:2, 2]
    # C0,2 = A00*B02 + A01*B12
    # C1,2 = A10*B02 + A11*B12
    # This is a 2x2x1 multiplication. Rank 4.
    # We can use schoolbook:
    # T8 = A00 * B02 -> C0,2 += T8
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1
    col += 1
    # T9 = A01 * B12 -> C0,2 += T9
    U[1][col] = 1; V[5][col] = 1; W[4][col] = 1
    col += 1
    # T10 = A10 * B02 -> C1,2 += T10
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1
    col += 1
    # T11 = A11 * B12 -> C1,2 += T11
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1
    col += 1
    
    # Terms for C[0:2, :] from A[0:2, 2] * B[2, :]
    # C0,0 = A02*B20, C0,1 = A02*B21, C0,2 = A02*B22
    # C1,0 = A12*B20, C1,1 = A12*B21, C1,2 = A12*B22
    # This is a 2x1x3 multiplication. Rank 6.
    # We can use schoolbook:
    # T12 = A02 * B20 -> C0,0 += T12
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1
    col += 1
    # T13 = A02 * B21 -> C0,1 += T13
    U[2][col] = 1; V[7][col] = 1; W[1][col] = 1
    col += 1
    # T14 = A02 * B22 -> C0,2 += T14
    U[2][col] = 1; V[8][col] = 1; W[4][col] = 1
    col += 1
    # T15 = A12 * B20 -> C1,0 += T15
    U[5][col] = 1; V[6][col] = 1; W[3][col] = 1
    col += 1
    # T16 = A12 * B21 -> C1,1 += T16
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1
    col += 1
    # T17 = A12 * B22 -> C1,2 += T17
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
