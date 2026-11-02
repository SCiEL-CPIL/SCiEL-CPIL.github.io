import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55: 5 * 7 (Strassen 2x2x2) + 20 (2x1x10)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of 2x2x2
    # For each block b (0..4):
    #   A sub-block: rows 0,1 of A (2x2), using cols 0,1 of the 3-col A
    #   B sub-block: rows 0,1 of B (2x2), using cols k0, k1 of the 10-col B
    #   C sub-block: rows 0,1 of C (2x2), using cols k0, k1 of the 10-col C
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Standard Strassen for 2x2 * 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0 * m + 0][col] = 1   # A00
        U[1 * m + 1][col] = 1   # A11
        V[0 * p + k0][col] = 1  # B00
        V[1 * p + k1][col] = 1  # B11
        W[k0 * n + 0][col] = 1  # C00
        W[k1 * n + 1][col] = 1  # C11
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1 * m + 0][col] = 1   # A10
        U[1 * m + 1][col] = 1   # A11
        V[0 * p + k0][col] = 1  # B00
        W[k1 * n + 0][col] = 1  # C10
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0 * m + 0][col] = 1   # A00
        V[0 * p + k1][col] = 1  # B01
        V[1 * p + k1][col] = -1 # B11
        W[k0 * n + 1][col] = 1  # C01
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[1 * m + 1][col] = 1   # A11
        V[1 * p + k0][col] = 1  # B10
        V[0 * p + k0][col] = -1 # B00
        W[k1 * n + 0][col] = 1  # C10
        W[k0 * n + 0][col] = -1 # C00
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0 * m + 0][col] = 1   # A00
        U[0 * m + 1][col] = 1   # A01
        V[1 * p + k1][col] = 1  # B11
        W[k0 * n + 0][col] = 1  # C00
        W[k0 * n + 1][col] = 1  # C01
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[1 * m + 0][col] = 1   # A10
        U[0 * m + 0][col] = -1  # A00
        V[0 * p + k0][col] = 1  # B00
        V[0 * p + k1][col] = 1  # B01
        W[k1 * n + 1][col] = 1  # C11
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + 1][col] = 1   # A01
        U[1 * m + 1][col] = -1  # A11
        V[1 * p + k0][col] = 1  # B10
        V[1 * p + k1][col] = 1  # B11
        W[k0 * n + 0][col] = -1 # C00
        W[k1 * n + 1][col] = 1  # C11
        col += 1

    # 2x1x10 part: A col 2 (index 2), B row 2 (index 2)
    # This is just standard multiplication for the remaining dimension
    # A[i,2] * B[2,k] contributes to C[i,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
