"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Decompose 3x4 * 4x4 into:
    #   A = [A1 (2x4) ; A2 (1x4)]
    #   B is 4x4
    #   C = A1*B + A2*B
    #
    # For A1 (2x4) * B (4x4):
    #   Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    #   A1 = [A1L (2x2) | A1R (2x2)]
    #   C1 = A1L*B11 + A1R*B21  (left half of C1)
    #   C2 = A1L*B12 + A1R*B22  (right half of C1)
    #   Use Strassen for each 2x2*2x2 multiplication (rank 7 each)
    #   Total rank for A1 part: 4 * 7 = 28
    #
    # For A2 (1x4) * B (4x4):
    #   Naive: each output element is a dot product of length 4
    #   4 outputs * 4 inputs = 16 rank-1 terms
    #   Total rank for A2 part: 16
    #
    # Total rank: 28 + 16 = 44
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # C11 = M1 - M2 + M4 + M5
    # C12 = M3 + M2 - M5 - M7
    # C21 = M1 + M4 - M3 - M6
    # C22 = M2 - M1 + M4 + M7
    #
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    strassen_coeffs = [
        # (M_idx, a_coeff, b_coeff, c_coeff)
        # Each entry: (m_index, a_sign, b_sign, c_sign) for output position
    ]
    
    def add_strassen_block(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add Strassen decomposition for a 2x2 * 2x2 block.
        a_row_offset: row offset in A (for 2x4, this is 0 or 2 for A1L, A1R)
        b_col_offset: column offset in B for the 2x2 block
        c_row_offset: row offset in C for the 2x2 result block
        c_col_offset: column offset in C for the 2x2 result block
        """
        nonlocal col
        # For 2x2 * 2x2, we need 7 terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Output: C11 = M1 - M2 + M4 + M5
        #         C12 = M3 + M2 - M5 - M7
        #         C21 = M1 + M4 - M3 - M6
        #         C22 = M2 - M1 + M4 + M7
        
        # M1: A rows 0,1 (A11, A22), B cols 0,1 (B11, B22)
        # A11 is at (a_row_offset+0, b_col_offset+0) -> but A is 2x4, so A1L and A1R
        # Wait, I need to be more careful with indexing.
        
        # For the 2x4 * 4x4 case:
        # A1 is 2x4, B is 4x4
        # We split A1 = [A1L (2x2) | A1R (2x2)]
        # B = [[B11 B12], [B21 B22]] where each is 2x2
        # 
        # C1_left = A1L * B11 + A1R * B21
        # C1_right = A1L * B12 + A1R * B22
        #
        # So we have 4 Strassen blocks:
        # 1. A1L (2x2) * B11 (2x2) -> contributes to C1_left
        # 2. A1R (2x2) * B21 (2x2) -> contributes to C1_left
        # 3. A1L (2x2) * B12 (2x2) -> contributes to C1_right
        # 4. A1R (2x2) * B22 (2x2) -> contributes to C1_right
        
        pass
    
    # Let me rebuild this more carefully.
    # U has 12 rows: A is 3x4, indexed as row i*m + j where i in [0,3), j in [0,4)
    # V has 16 rows: B is 4x4, indexed as row j*p + k where j in [0,4), k in [0,4)
    # W has 12 rows: C is 4x3 (p*n = 4*3), indexed as W[k*n+i] where k in [0,4), i in [0,3)
    
    # A1 = rows 0,1 of A (2x4): U indices 0..3 (i=0, j=0..3) and 4..7 (i=1, j=0..3)
    # A2 = row 2 of A (1x4): U indices 8..11 (i=2, j=0..3)
    
    # B is 4x4: V indices 0..15 (j=0..3, k=0..3)
    # B11 = B[0:2, 0:2]: V indices 0..3
    # B12 = B[0:2, 2:4]: V indices 4..7
    # B21 = B[2:4, 0:2]: V indices 8..11
    # B22 = B[2:4, 2:4]: V indices 12..15
    
    # C1 = rows 0,1 of C (4x2): W indices k*3+0 and k*3+1 for k=0..3
    # C2 = row 2 of C (4x1): W indices k*3+2 for k=0..3
    
    # Strassen for 2x2 * 2x2:
    # A block 2x2, B block 2x2, C block 2x2
    # For block (A_block, B_block, C_block):
    #   A_block rows: 2 rows, cols: 2 cols
    #   B_block rows: 2 rows, cols: 2 cols  
    #   C_block rows: 2 rows, cols: 2 cols
    
    # Let me define Strassen for a general 2x2 * 2x2:
    # Given A (2x2), B (2x2), compute C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M2 + M4 + M5
    # C01 = M3 + M2 - M5 - M7
    # C10 = M1 + M4 - M3 - M6
    # C11 = M2 - M1 + M4 + M7
    
    def add_strassen_2x2(a_rows, b_rows, c_rows, a_cols, b_cols, c_cols):
        """Add Strassen terms for a 2x2 * 2x2 multiplication.
        a_rows: 2 row indices in U (for A)
        a_cols: 2 col indices in A (but A is 4-wide, so these are which of the 4 cols)
        b_rows: 2 row indices in V (for B, these are j indices)
        b_cols: 2 col indices in B (k indices)
        c_rows: 2 row indices in W (for C, these are k indices)
        c_cols: 2 col indices in C (i indices)
        """
        nonlocal col
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        U[a_rows[0]][col] += 1
        U[a_rows[1]][col] += 1
        V[b_rows[0]*4 + b_cols[0]][col] += 1  # B00
        V[b_rows[1]*4 + b_cols[1]][col] += 1  # B11
        W[c_rows[0]*3 + c_cols[0]][col] += 1   # C00
        W[c_rows[0]*3 + c_cols[1]][col] += 1   # C01? No, wait.
        # Actually C00 gets +1, C10 gets +1
        W[c_rows[0]*3 + c_cols[0]][col] = 1
        W[c_rows[1]*3 + c_cols[0]][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_rows[1]][col] += 1
        U[a_rows[1]][col] += 1  # A11, but wait A10 and A11 are different
        # A10 is a_rows[1], a_cols[0]
        # A11 is a_rows[1], a_cols[1]
        U[a_rows[1]][col] = 1  # A10
        # Need to add A11 separately - but Strassen combines them
        # M2 = (A10+A11) * B00
        # So U gets A10 and A11, V gets B00
        U[a_rows[1]][col] = 1  # this is wrong, need to handle properly
        
        # Let me redo this more carefully.
        pass
    
    # I'm getting confused. Let me start over with a cleaner approach.
    
    rank = 44
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(12)]
    col = 0
    
    # Strassen 2x2 * 2x2: 7 terms
    # For each 2x2*2x2 block, specify:
    #   a00_u, a01_u, a10_u, a11_u: U row indices for A's 4 elements
    #   b00_v, b01_v, b10_v, b11_v: V row indices for B's 4 elements  
    #   c00_w, c01_w, c10_w, c11_w: W row indices for C's 4 elements
    
    def strassen_block(a_u, b_v, c_w):
        """Add 7 Strassen terms. a_u, b_v, c_w are lists of 4 indices each."""
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[a_u[0]][col] = 1
        U[a_u[3]][col] = 1
        V[b_v[0]][col] = 1
        V[b_v[3]][col] = 1
        W[c_w[0]][col] = 1
        W[c_w[2]][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_u[2]][col] = 1
        U[a_u[3]][col] = 1
        V[b_v[0]][col] = 1
        W[c_w[0]][col] = -1
        W[c_w[1]][col] = 1
        W[c_w[3]][col] = 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a_u[0]][col] = 1
        V[b_v[1]][col] = 1
        V[b_v[3]][col] = -1
        W[c_w[1]][col] = 1
        W[c_w[2]][col] = -1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a_u[3]][col] = 1
        V[b_v[2]][col] = 1
        V[b_v[0]][col] = -1
        W[c_w[0]][col] = 1
        W[c_w[2]][col] = 1
        W[c_w[3]][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a_u[0]][col] = 1
        U[a_u[1]][col] = 1
        V[b_v[3]][col] = 1
        W[c_w[0]][col] = 1
        W[c_w[1]][col] = -1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_u[2]][col] = 1
        U[a_u[0]][col] = -1
        V[b_v[0]][col] = 1
        V[b_v[1]][col] = 1
        W[c_w[2]][col] = -1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_u[1]][col] = 1
        U[a_u[3]][col] = -1
        V[b_v[2]][col] = 1
        V[b_v[3]][col] = 1
        W[c_w[1]][col] = -1
        W[c_w[3]][col] = 1
        col += 1
    
    # A1L = A[0:2, 0:2], A1R = A[0:2, 2:4]
    # A1L: U rows 0,1 (i=0,j=0) and 2,3 (i=1,j=0) -> a_u = [0, 1, 4, 5]
    # Wait: U row = i*m + j = i*4 + j
    # A1L: i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 4; i=1,j=1 -> 5
    # A1R: i=0,j=2 -> 2; i=0,j=3 -> 3; i=1,j=2 -> 6; i=1,j=3 -> 7
    
    # B11: j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 4; j=1,k=1 -> 5
    # B12: j=0,k=2 -> 2; j=0,k=3 -> 3; j=1,k=2 -> 6; j=1,k=3 -> 7
    # B21: j=2,k=0 -> 8; j=2,k=1 -> 9; j=3,k=0 -> 12; j=3,k=1 -> 13
    # B22: j=2,k=2 -> 10; j=2,k=3 -> 11; j=3,k=2 -> 14; j=3,k=3 -> 15
    
    # C1_left (k=0,1; i=0,1): W[k*3+i]
    #   C11: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 3; k=1,i=1 -> 4
    # C1_right (k=2,3; i=0,1): 
    #   k=2,i=0 -> 6; k=2,i=1 -> 7; k=3,i=0 -> 9; k=3,i=1 -> 10
    
    # Block 1: A1L * B11 -> C1_left
    # A1L: [0,1,4,5], B11: [0,1,4,5], C1_left: [0,1,3,4]
    strassen_block([0,1,4,5], [0,1,4,5], [0,1,3,4])
    
    # Block 2: A1R * B21 -> C1_left
    # A1R: [2,3,6,7], B21: [8,9,12,13], C1_left: [0,1,3,4]
    strassen_block([2,3,6,7], [8,9,12,13], [0,1,3,4])
    
    # Block 3: A1L * B12 -> C1_right
    # A1L: [0,1,4,5], B12: [2,3,6,7], C1_right: [6,7,9,10]
    strassen_block([0,1,4,5], [2,3,6,7], [6,7,9,10])
    
    # Block 4: A1R * B22 -> C1_right
    # A1R: [2,3,6,7], B22: [10,11,14,15], C1_right: [6,7,9,10]
    strassen_block([2,3,6,7], [10,11,14,15], [6,7,9,10])
    
    # Now A2 (row 2, 1x4) * B (4x4) -> C2 (4x1)
    # Naive: 16 terms
    # C2[k] = sum_j A2[j] * B[j,k]
    # A2[j]: U row 8+j
    # B[j,k]: V row j*4+k
    # C2[k]: W row k*3+2
    for k in range(4):
        for j in range(4):
            U[8+j][col] = 1
            V[j*4+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    assert col == 44, f"Expected 44, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
