import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Group 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1 (i), 0,1 (j) -> U index i*m+j
    # B indices: 0,1 (j), 0,1 (k) -> V index j*p+k
    # W indices: 0,1 (k), 0,1 (i) -> W index k*n+i
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1) # A00 B00
    add_term(0, 1, 0, 1) # A00 B10 ? No, B01 is j=0, k=1.
    
    # Let's use a helper for Strassen
    def add_strassen(a_row1, a_col1, a_row2, a_col2, b_row1, b_col1, b_row2, b_col2, offset):
        # A is a_row1..a_row2, a_col1..a_col2
        # B is b_row1..b_row2, b_col1..b_col2
        # Returns 7 terms added to U, V, W
        pass

    # It's easier to just list the 7 Strassen terms explicitly for the first block
    # A00, A01, A10, A11
    # B00, B01, B10, B11
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] += 1; V[0][col] += 1; W[0][col] += 1 # A00 B00
    U[0][col] += 1; V[1][col] += 1; W[1][col] += 1 # A00 B10 (j=1, k=0) -> V index 1*6+0 = 6?
    # Wait, V index is j*p+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 6
    # j=1, k=1 -> 7
    
    # Let's rewrite carefully.
    # U index: i*3 + j
    # V index: j*6 + k
    # W index: k*2 + i

    # M1 = (A00+A11)(B00+B11)
    # Terms: A00B00, A00B11, A11B00, A11B11
    # We add these as separate rank-1 terms? No, Strassen combines them into ONE product.
    # The product is (A00+A11) * (B00+B11).
    # So U has A00+A11, V has B00+B11, W has 1 for the output?
    # No, the output is a linear combination of M1..M7.
    # So we add 7 columns to U, V, W.
    # For each column, U has the linear combo of A, V has linear combo of B, W has the coefficient for the output.
    
    # Let's implement a generic add_strassen_block
    def add_strassen_block(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2):
        nonlocal col
        # A submatrix rows a_r1..a_r2, cols a_c1..a_c2
        # B submatrix rows b_r1..b_r2, cols b_c1..b_c2
        # Outputs C submatrix rows a_r1..a_r2, cols b_c1..b_c2
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Note: 1-based in formula, 0-based in code
        # A11 = A[a_r1, a_c1], A12 = A[a_r1, a_c2]
        # A21 = A[a_r2, a_c1], A22 = A[a_r2, a_c2]
        
        # B11 = B[b_r1, b_c1], B12 = B[b_r1, b_c2]
        # B21 = B[b_r2, b_c1], B22 = B[b_r2, b_c2]
        
        # M1
        U[a_r1*3 + a_c1][col] += 1
        U[a_r2*3 + a_c2][col] += 1
        V[b_r1*6 + b_c1][col] += 1
        V[b_r2*6 + b_c2][col] += 1
        # W coefficients for M1:
        # C11 += M1, C12 -= M1? No.
        # C11 = M1 - M3 + M4 - M7
        # C12 = M5 + M7
        # C21 = M1 + M2
        # C22 = M2 - M4 + M5 - M6
        
        W[b_c1*2 + a_r1][col] += 1  # C11
        W[b_c2*2 + a_r1][col] += 0   # C12 (no M1)
        W[b_c1*2 + a_r2][col] += 1   # C21
        W[b_c2*2 + a_r2][col] += 0   # C22
        col += 1
        
        # M2
        U[a_r2*3 + a_c1][col] += 1
        U[a_r2*3 + a_c2][col] += 1
        V[b_r1*6 + b_c1][col] += 1
        W[b_c1*2 + a_r1][col] += 0
        W[b_c2*2 + a_r1][col] += 0
        W[b_c1*2 + a_r2][col] += 1   # C21
        W[b_c2*2 + a_r2][col] += 1   # C22
        col += 1
        
        # M3
        U[a_r1*3 + a_c1][col] += 1
        V[b_r1*6 + b_c2][col] += 1
        V[b_r2*6 + b_c2][col] += -1
        W[b_c1*2 + a_r1][col] += -1  # C11
        W[b_c2*2 + a_r1][col] += 0
        W[b_c1*2 + a_r2][col] += 0
        W[b_c2*2 + a_r2][col] += 0
        col += 1
        
        # M4
        U[a_r2*3 + a_c2][col] += 1
        V[b_r2*6 + b_c1][col] += 1
        V[b_r1*6 + b_c1][col] += -1
        W[b_c1*2 + a_r1][col] += 1   # C11
        W[b_c2*2 + a_r1][col] += 0
        W[b_c1*2 + a_r2][col] += 0
        W[b_c2*2 + a_r2][col] += -1  # C22
        col += 1
        
        # M5
        U[a_r1*3 + a_c1][col] += 1
        U[a_r1*3 + a_c2][col] += 1
        V[b_r2*6 + b_c2][col] += 1
        W[b_c1*2 + a_r1][col] += 0
        W[b_c2*2 + a_r1][col] += 1   # C12
        W[b_c1*2 + a_r2][col] += 0
        W[b_c2*2 + a_r2][col] += 1   # C22
        col += 1
        
        # M6
        U[a_r2*3 + a_c1][col] += 1
        U[a_r1*3 + a_c1][col] += -1
        V[b_r1*6 + b_c1][col] += 1
        V[b_r1*6 + b_c2][col] += 1
        W[b_c1*2 + a_r1][col] += 0
        W[b_c2*2 + a_r1][col] += 0
        W[b_c1*2 + a_r2][col] += 0
        W[b_c2*2 + a_r2][col] += -1  # C22
        col += 1
        
        # M7
        U[a_r1*3 + a_c2][col] += 1
        U[a_r2*3 + a_c2][col] += -1
        V[b_r2*6 + b_c1][col] += 1
        V[b_r2*6 + b_c2][col] += 1
        W[b_c1*2 + a_r1][col] += -1  # C11
        W[b_c2*2 + a_r1][col] += 1   # C12
        W[b_c1*2 + a_r2][col] += 0
        W[b_c2*2 + a_r2][col] += 0
        col += 1

    # Group 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 1, 1, 0, 0, 1, 1)
    
    # Group 3: A[0:2, 0:2] * B[0:2, 2:4] and B[0:2, 4:6]
    # B rows 0,1. Cols 2,3 for first block.
    add_strassen_block(0, 0, 1, 1, 0, 2, 1, 3)
    # B rows 0,1. Cols 4,5 for second block.
    add_strassen_block(0, 0, 1, 1, 0, 4, 1, 5)
    
    # Group 2: j=2 (index 2)
    # A[0:2, 2] * B[2:3, 0:6]
    # A indices: 0*3+2=2, 1*3+2=5
    # B indices: 2*6+k = 12+k
    # W indices: k*2+i
    
    for k in range(6):
        # A02 * B2k -> C0k
        U[2][col] += 1
        V[12+k][col] += 1
        W[k*2 + 0][col] += 1
        col += 1
        
        # A12 * B2k -> C1k
        U[5][col] += 1
        V[12+k][col] += 1
        W[k*2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
