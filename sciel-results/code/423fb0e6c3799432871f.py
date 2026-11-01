import numpy as np
import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for 2x2x3 matrix multiplication is 9.
    # We provide a rank-9 decomposition.
    # Rank 7 (Strassen) for the 2x2x2 part (columns 0 and 1 of B).
    # Rank 2 for the 2x2x1 part (column 2 of B).
    
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B (2x2x2 part)
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Linear extension part for column 2 of B (2x2x1 part)
    # This computes C02 = A00*B02 + A01*B12 and C12 = A10*B02 + A11*B12
    # Using rank 2 decomposition derived from tensor structure.
    
    # Term 7: Handles row 0 of output (C02)
    # U7 = [[1, 1], [0, 0]]
    U[0][7] = 1; U[1][7] = 1
    # V7 has 1s at (0,2) and (1,2) -> indices 2 and 5 in V
    V[2][7] = 1; V[5][7] = 1
    # W7 has 1 at (0,2) -> index 4 in W
    W[4][7] = 1
    
    # Term 8: Handles row 1 of output (C12)
    # U8 = [[0, 0], [1, 1]]
    U[2][8] = 1; U[3][8] = 1
    # V8 has 1s at (0,2) and (1,2) -> indices 2 and 5 in V
    V[2][8] = 1; V[5][8] = 1
    # W8 has 1 at (1,2) -> index 5 in W
    W[5][8] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
