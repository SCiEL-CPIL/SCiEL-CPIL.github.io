import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_2 B_2 (2x1x7)
    # A_2 is col 2 of A (indices 2, 5 in U)
    # B_2 is row 2 of B (indices 14..20 in V)
    # C is all k, i
    for k in range(7):
        # Term 1: A_0,2 * B_2,k -> C_k,0
        U[0 * 3 + 2][col] = 1
        V[2 * 7 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # Term 2: A_1,2 * B_2,k -> C_k,1
        U[1 * 3 + 2][col] = 1
        V[2 * 7 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
    # Part 2: A_1 B_1 (2x2x7)
    # A_1 is cols 0,1 of A (indices 0,1,3,4 in U)
    # B_1 is rows 0,1 of B (indices 0..13 in V)
    # Split k into 0,1; 2,3; 4,5; 6
    
    # Strassen blocks for k=0,1; 2,3; 4,5
    for start_k in [0, 2, 4]:
        k0 = start_k
        k1 = start_k + 1
        
        # M1 = A00 * (B00 + B11)
        U[0 * 3 + 0][col] = 1
        V[0 * 7 + k0][col] = 1
        V[1 * 7 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1  # C00
        W[k1 * 2 + 1][col] = 1  # C11
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1 * 3 + 0][col] = 1
        U[1 * 3 + 1][col] = 1
        V[0 * 7 + k0][col] = 1
        W[k0 * 2 + 1][col] = 1  # C10
        W[k1 * 2 + 1][col] = 1  # C11
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0 * 3 + 0][col] = 1
        V[0 * 7 + k1][col] = 1
        V[1 * 7 + k1][col] = -1
        W[k1 * 2 + 0][col] = 1  # C01
        W[k1 * 2 + 1][col] = -1 # C11
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[1 * 3 + 1][col] = 1
        V[1 * 7 + k0][col] = 1
        V[0 * 7 + k0][col] = -1
        W[k0 * 2 + 0][col] = 1  # C00
        W[k0 * 2 + 1][col] = 1  # C10
        col += 1
        
        # M5 = (A00 + A11) * (B00 + B01)
        U[0 * 3 + 0][col] = 1
        U[1 * 3 + 1][col] = 1
        V[0 * 7 + k0][col] = 1
        V[0 * 7 + k1][col] = 1
        W[k0 * 2 + 0][col] = -1 # C00
        W[k1 * 2 + 0][col] = 1  # C01
        col += 1
        
        # M6 = (A10 - A00) * (B10 + B11)
        U[1 * 3 + 0][col] = 1
        U[0 * 3 + 0][col] = -1
        V[1 * 7 + k0][col] = 1
        V[1 * 7 + k1][col] = 1
        W[k1 * 2 + 1][col] = -1 # C11
        col += 1
        
        # M7 = (A01 - A11) * (B01 + B11)
        U[0 * 3 + 1][col] = 1
        U[1 * 3 + 1][col] = -1
        V[0 * 7 + k1][col] = 1
        V[1 * 7 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1  # C00
        W[k1 * 2 + 1][col] = 1  # C11
        col += 1

    # Block for k=6 (2x2x1)
    k0 = 6
    # A00 * B00 -> C00
    U[0 * 3 + 0][col] = 1
    V[0 * 7 + k0][col] = 1
    W[k0 * 2 + 0][col] = 1
    col += 1
    # A01 * B10 -> C00
    U[0 * 3 + 1][col] = 1
    V[1 * 7 + k0][col] = 1
    W[k0 * 2 + 0][col] = 1
    col += 1
    # A10 * B00 -> C10
    U[1 * 3 + 0][col] = 1
    V[0 * 7 + k0][col] = 1
    W[k0 * 2 + 1][col] = 1
    col += 1
    # A11 * B10 -> C10
    U[1 * 3 + 1][col] = 1
    V[1 * 7 + k0][col] = 1
    W[k0 * 2 + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
