"""Rewritten program using a block-structured construction for matrix multiplication tensor."""
DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 27 construction:
    # Split A into 2x2 (cols 0,1) and 2x1 (col 2)
    # Split B into 2x6 (rows 0,1) and 1x6 (row 2)
    # C = A_left @ B_top + A_right @ B_bottom
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_left (2x2) @ B_top (2x6)
    # B_top is split into 3 blocks of 2x2 (cols 0-1, 2-3, 4-5)
    # Strassen for 2x2x2 has rank 7
    for block_idx in range(3):
        # B block cols: 2*block_idx, 2*block_idx+1
        b_col_start = 2 * block_idx
        
        # Strassen terms for this block
        # M1 = (A00+A11)(B00+B11)
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {b_col_start, b_col_start+1}
        # C indices: i in {0,1}, k in {b_col_start, b_col_start+1}
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11 (with coeffs)
        
        # We will add 7 terms
        
        # T1: (A00+A11) * (B00+B11)
        U[0*2+0][col] = 1 # A00
        U[1*2+1][col] = 1 # A11
        V[0*6+b_col_start][col] = 1 # B00
        V[1*6+b_col_start+1][col] = 1 # B11
        # C contributions:
        # C00 += M1
        # C01 += M1
        # C10 += M1
        # C11 += M1
        # W index k*n + i
        W[(b_col_start)*2+0][col] = 1 # C00
        W[(b_col_start+1)*2+0][col] = 1 # C01
        W[(b_col_start)*2+1][col] = 1 # C10
        W[(b_col_start+1)*2+1][col] = 1 # C11
        col += 1
        
        # T2: (A10+A11) * B00
        U[1*2+0][col] = 1 # A10
        U[1*2+1][col] = 1 # A11
        V[0*6+b_col_start][col] = 1 # B00
        # C10 += M2, C11 -= M2
        W[(b_col_start)*2+1][col] = 1 # C10
        W[(b_col_start+1)*2+1][col] = -1 # C11
        col += 1
        
        # T3: A00 * (B01-B11)
        U[0*2+0][col] = 1 # A00
        V[0*6+b_col_start+1][col] = 1 # B01
        V[1*6+b_col_start+1][col] = -1 # B11
        # C01 += M3, C11 += M3
        W[(b_col_start+1)*2+0][col] = 1 # C01
        W[(b_col_start+1)*2+1][col] = 1 # C11
        col += 1
        
        # T4: A11 * (B10-B00)
        U[1*2+1][col] = 1 # A11
        V[1*6+b_col_start][col] = 1 # B10
        V[0*6+b_col_start][col] = -1 # B00
        # C10 += M4, C11 += M4
        W[(b_col_start)*2+1][col] = 1 # C10
        W[(b_col_start+1)*2+1][col] = 1 # C11
        col += 1
        
        # T5: (A00+A01) * B11
        U[0*2+0][col] = 1 # A00
        U[0*2+1][col] = 1 # A01
        V[1*6+b_col_start+1][col] = 1 # B11
        # C00 -= M5, C01 += M5
        W[(b_col_start)*2+0][col] = -1 # C00
        W[(b_col_start+1)*2+0][col] = 1 # C01
        col += 1
        
        # T6: (A10-A00) * (B00+B10)
        U[1*2+0][col] = 1 # A10
        U[0*2+0][col] = -1 # A00
        V[0*6+b_col_start][col] = 1 # B00
        V[1*6+b_col_start][col] = 1 # B10
        # C11 += M6
        W[(b_col_start+1)*2+1][col] = 1 # C11
        col += 1
        
        # T7: (A01-A11) * (B01+B11)
        U[0*2+1][col] = 1 # A01
        U[1*2+1][col] = -1 # A11
        V[0*6+b_col_start+1][col] = 1 # B01
        V[1*6+b_col_start+1][col] = 1 # B11
        # C00 += M7
        W[(b_col_start)*2+0][col] = 1 # C00
        col += 1

    # Part 2: A_right (2x1) @ B_bottom (1x6)
    # A_right is col 2 of A. B_bottom is row 2 of B.
    # C_i,k = A_2,2 * B_2,k
    for k in range(6):
        # Term: A_2,2 * B_2,k
        U[2*2+2][col] = 1 # A[2,2] -> index 2*2+2 = 6? 
        # Wait, A is 2x3. Index i*m+j.
        # i=0,1. j=0,1,2.
        # A_right corresponds to j=2.
        # U index for A[0,2] is 0*3+2 = 2.
        # U index for A[1,2] is 1*3+2 = 5.
        # We need two terms? No, it's a single product A_2,2 * B_2,k for each C?
        # C[0,k] = A[0,2] * B[2,k]
        # C[1,k] = A[1,2] * B[2,k]
        # These are separate scalar multiplications.
        # So we need 2 terms per k?
        # Term 1: A[0,2] * B[2,k] -> contributes to C[0,k]
        # Term 2: A[1,2] * B[2,k] -> contributes to C[1,k]
        
        # Term for C[0,k]
        U[0*3+2][col] = 1 # A[0,2]
        V[2*6+k][col] = 1 # B[2,k]
        W[k*2+0][col] = 1 # C[0,k]
        col += 1
        
        # Term for C[1,k]
        U[1*3+2][col] = 1 # A[1,2]
        V[2*6+k][col] = 1 # B[2,k]
        W[k*2+1][col] = 1 # C[1,k]
        col += 1

    return {"U": U, "V": V, "W": W}
