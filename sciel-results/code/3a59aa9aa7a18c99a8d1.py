import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for Rank 88
    # U: 12 x 88
    # V: 32 x 88
    # W: 24 x 88
    
    U = np.zeros((n*m, 88), dtype=float)
    V = np.zeros((m*p, 88), dtype=float)
    W = np.zeros((p*n, 88), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, coeff):
        nonlocal col
        U[i*m + j, col] = coeff
        V[j*p + k, col] = coeff
        W[k*n + i, col] = coeff
        col += 1
        
    # Part 1: A_L (cols 0,1) * B_T (rows 0,1)
    # Local A is 3x2, Local B is 2x8
    # Split B into 4 blocks of 2x2
    for block_idx in range(4):
        # Block k indices: 2*block_idx, 2*block_idx+1
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # A_top (row 0) * B_block
        # A_top is global i=0, local j=0,1
        # B_block is global j=0,1, local k=k0,k1
        
        # Rank 4 terms for 1x2 * 2x2
        # C = [a0*b00 + a1*b10, a0*b01 + a1*b11]
        # a0 is A[0,0], a1 is A[0,1]
        # b00 is B[0,k0], b01 is B[0,k1]
        # b10 is B[1,k0], b11 is B[1,k1]
        
        # Term 1: a0 * b00 -> C[0, k0]
        add_term(0, 0, k0, 1)
        # Term 2: a0 * b01 -> C[0, k1]
        add_term(0, 0, k1, 1)
        # Term 3: a1 * b10 -> C[0, k0]
        add_term(0, 1, k0, 1)
        # Term 4: a1 * b11 -> C[0, k1]
        add_term(0, 1, k1, 1)
        
        # A_bot (rows 1,2) * B_block
        # A_bot is 2x2. Global i=1,2. Local j=0,1.
        # B_block is 2x2. Global j=0,1. Local k=k0,k1.
        # Strassen Rank 7
        
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[1,0], A22 = A[2,1]
        # B11 = B[0,k0], B22 = B[1,k1]
        
        # We need to express linear combinations.
        # The tensor decomposition allows linear combinations in U, V, W.
        # U[i, c] can be sum of coeffs.
        
        # Let's define the Strassen terms explicitly.
        # M1
        # U parts: A11 (i=1, j=0) and A22 (i=2, j=1)
        # V parts: B11 (j=0, k=k0) and B22 (j=1, k=k1)
        # W parts: 
        # C11 += M1 -> W[k0*n + 1] += 1
        # C12 += M1 -> W[k1*n + 1] += 0? No.
        # Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add 7 terms to the global U, V, W.
        
        # Term M1
        # U: A11 (1,0) coeff 1, A22 (2,1) coeff 1
        # V: B11 (0,k0) coeff 1, B22 (1,k1) coeff 1
        # W: C11 (k0,1) coeff 1, C22 (k1,2) coeff 1
        # Note: C11 is global i=1, k=k0. C22 is global i=2, k=k1.
        
        U[1*m+0, col] = 1
        U[2*m+1, col] = 1
        V[0*p+k0, col] = 1
        V[1*p+k1, col] = 1
        W[k0*n+1, col] = 1
        W[k1*n+2, col] = 1
        col += 1
        
        # Term M2 = (A21+A22)B11
        # A21 = A[2,0], A22 = A[2,1]
        # B11 = B[0,k0]
        # C21 += M2, C22 -= M2
        U[2*m+0, col] = 1
        U[2*m+1, col] = 1
        V[0*p+k0, col] = 1
        W[k0*n+2, col] = 1
        W[k1*n+2, col] = -1
        col += 1
        
        # Term M3 = A11(B21-B22)
        # A11 = A[1,0]
        # B21 = B[1,k0], B22 = B[1,k1]
        # C12 += M3, C22 += M3
        U[1*m+0, col] = 1
        V[1*p+k0, col] = 1
        V[1*p+k1, col] = -1
        W[k1*n+1, col] = 1
        W[k1*n+2, col] = 1
        col += 1
        
        # Term M4 = A22(B12-B11)
        # A22 = A[2,1]
        # B12 = B[0,k1], B11 = B[0,k0]
        # C11 += M4, C21 += M4
        U[2*m+1, col] = 1
        V[0*p+k1, col] = 1
        V[0*p+k0, col] = -1
        W[k0*n+1, col] = 1
        W[k0*n+2, col] = 1
        col += 1
        
        # Term M5 = (A11+A12)B22
        # A11 = A[1,0], A12 = A[1,1]
        # B22 = B[1,k1]
        # C11 -= M5, C12 += M5
        U[1*m+0, col] = 1
        U[1*m+1, col] = 1
        V[1*p+k1, col] = 1
        W[k0*n+1, col] = -1
        W[k1*n+1, col] = 1
        col += 1
        
        # Term M6 = (A21-A11)(B11+B12)
        # A21 = A[2,0], A11 = A[1,0]
        # B11 = B[0,k0], B12 = B[0,k1]
        # C22 += M6
        U[2*m+0, col] = 1
        U[1*m+0, col] = -1
        V[0*p+k0, col] = 1
        V[0*p+k1, col] = 1
        W[k1*n+2, col] = 1
        col += 1
        
        # Term M7 = (A12-A22)(B21+B22)
        # A12 = A[1,1], A22 = A[2,1]
        # B21 = B[1,k0], B22 = B[1,k1]
        # C11 += M7
        U[1*m+1, col] = 1
        U[2*m+1, col] = -1
        V[1*p+k0, col] = 1
        V[1*p+k1, col] = 1
        W[k0*n+1, col] = 1
        col += 1

    # Part 2: A_R (cols 2,3) * B_B (rows 2,3)
    # Local A is 3x2 (cols 2,3 of global A). Global j=2,3.
    # Local B is 2x8 (rows 2,3 of global B). Global j=2,3.
    
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # A_top (row 0) * B_block
        # Global i=0. Global j=2,3.
        # Global B j=2,3.
        
        # Term 1: A[0,2] * B[2,k0]
        add_term(0, 2, k0, 1)
        # Term 2: A[0,2] * B[2,k1]
        add_term(0, 2, k1, 1)
        # Term 3: A[0,3] * B[3,k0]
        add_term(0, 3, k0, 1)
        # Term 4: A[0,3] * B[3,k1]
        add_term(0, 3, k1, 1)
        
        # A_bot (rows 1,2) * B_block
        # Global i=1,2. Global j=2,3.
        # Global B j=2,3.
        
        # Strassen mapping:
        # A11 = A[1,2], A12 = A[1,3]
        # A21 = A[2,2], A22 = A[2,3]
        # B11 = B[2,k0], B12 = B[2,k1]
        # B21 = B[3,k0], B22 = B[3,k1]
        
        # M1
        U[1*m+2, col] = 1
        U[2*m+3, col] = 1
        V[2*p+k0, col] = 1
        V[3*p+k1, col] = 1
        W[k0*n+1, col] = 1
        W[k1*n+2, col] = 1
        col += 1
        
        # M2
        U[2*m+2, col] = 1
        U[2*m+3, col] = 1
        V[2*p+k0, col] = 1
        W[k0*n+2, col] = 1
        W[k1*n+2, col] = -1
        col += 1
        
        # M3
        U[1*m+2, col] = 1
        V[3*p+k0, col] = 1
        V[3*p+k1, col] = -1
        W[k1*n+1, col] = 1
        W[k1*n+2, col] = 1
        col += 1
        
        # M4
        U[2*m+3, col] = 1
        V[2*p+k1, col] = 1
        V[2*p+k0, col] = -1
        W[k0*n+1, col] = 1
        W[k0*n+2, col] = 1
        col += 1
        
        # M5
        U[1*m+2, col] = 1
        U[1*m+3, col] = 1
        V[3*p+k1, col] = 1
        W[k0*n+1, col] = -1
        W[k1*n+1, col] = 1
        col += 1
        
        # M6
        U[2*m+2, col] = 1
        U[1*m+2, col] = -1
        V[2*p+k0, col] = 1
        V[2*p+k1, col] = 1
        W[k1*n+2, col] = 1
        col += 1
        
        # M7
        U[1*m+3, col] = 1
        U[2*m+3, col] = -1
        V[3*p+k0, col] = 1
        V[3*p+k1, col] = 1
        W[k0*n+1, col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
