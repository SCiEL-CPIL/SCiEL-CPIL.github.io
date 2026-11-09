"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    # We'll build rank-44 decomposition:
    # - Rows 0,1 of A (2x4 block) with B (4x4) -> 2x4 output: rank 28 via Strassen-like
    # - Row 2 of A (1x4 block) with B (4x4) -> 1x4 output: rank 16 (optimal)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44, W[k*n+i, c]
    
    # Part 1: 1x4 x 4x4 -> 1x4 (row 2 of A), rank 16
    # A[2, j] * B[j, k] for j,k in 0..3
    # Output: C[2, k] for k in 0..3
    # W index: k*n + i = k*3 + 2
    col = 0
    for j in range(4):
        for k in range(4):
            U[2 * m + j][col] = 1  # row 2, col j of A
            V[j * p + k][col] = 1  # row j, col k of B
            W[k * n + 2][col] = 1  # output row k, col 2 -> index k*3+2
            col += 1
    # col is now 16
    
    # Part 2: 2x4 x 4x4 -> 2x4 (rows 0,1 of A), rank 28
    # We need to decompose this. Let's use a known 2x4x4 rank-28 decomposition.
    # Split: A = [A_L, A_R] where A_L is 2x2 (cols 0,1), A_R is 2x2 (cols 2,3)
    # B = [[B_TL, B_TR], [B_BL, B_BR]] each 2x2
    # C_L (2x2) = A_L * B_TL + A_R * B_BL
    # C_R (2x2) = A_L * B_TR + A_R * B_BR
    #
    # Each 2x2x2 product can be done with Strassen (rank 7).
    # C_L involves products: A_L*B_TL and A_R*B_BL
    # C_R involves products: A_L*B_TR and A_R*B_BR
    #
    # Total: 4 separate 2x2x2 Strassen decompositions = 4*7 = 28 terms.
    # But we can potentially share... let's just do 4 independent Strassen decomps.
    
    def strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
        """
        Decompose C[c_rows, c_cols] += A[a_rows, a_cols] * B[b_rows, b_cols]
        using Strassen's 7 products.
        a_rows: 2 indices for A rows
        a_cols: 2 indices for A cols  
        b_rows: 2 indices for B rows
        b_cols: 2 indices for B cols
        c_rows: 2 indices for C rows
        c_cols: 2 indices for C cols
        Returns list of (U_entry, V_entry, W_entry) as (row, col) pairs and the scalar.
        """
        # A submatrix: A[a_rows[0], a_cols[0:2]] and A[a_rows[1], a_cols[0:2]]
        # B submatrix: B[b_rows[0:2], b_cols[0:2]]
        # C submatrix: C[c_rows[0:2], c_cols[0:2]]
        
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        c00 = (c_rows[0], c_cols[0])
        c01 = (c_rows[0], c_cols[1])
        c10 = (c_rows[1], c_cols[0])
        c11 = (c_rows[1], c_cols[1])
        
        # Strassen 7 products:
        # M1 = (A00-A11)*(B11-B00)
        # M2 = (A00+A01)*B11
        # M3 = A00*(B00-B11)
        # M4 = A11*(B00+B11)
        # M5 = (A01-A11)*(B00+B10)
        # M6 = (A00+A11)*(B00-B01)
        # M7 = (A01-A00)*(B10+B11)
        #
        # C00 = M1 + M4 - M2 + M6
        # C01 = M2 + M3
        # C10 = M5 + M4
        # C11 = M1 - M3 + M5 + M7
        
        entries = []
        
        # M1 = (A00-A11)*(B11-B00)
        # U: A00 coeff +1, A11 coeff -1
        # V: B11 coeff +1, B00 coeff -1
        # C: C00 +1, C11 +1
        entries.append([
            (a00, +1), (a11, -1),
            (b11, +1), (b00, -1),
            (c00, +1), (c11, +1)
        ])
        # M2 = (A00+A01)*B11
        entries.append([
            (a00, +1), (a01, +1),
            (b11, +1),
            (c00, -1), (c01, +1)
        ])
        # M3 = A00*(B00-B11)
        entries.append([
            (a00, +1),
            (b00, +1), (b11, -1),
            (c01, +1), (c11, -1)
        ])
        # M4 = A11*(B00+B11)
        entries.append([
            (a11, +1),
            (b00, +1), (b11, +1),
            (c00, +1), (c10, +1)
        ])
        # M5 = (A01-A11)*(B00+B10)
        entries.append([
            (a01, +1), (a11, -1),
            (b00, +1), (b10, +1),
            (c10, +1), (c11, +1)
        ])
        # M6 = (A00+A11)*(B00-B01)
        entries.append([
            (a00, +1), (a11, +1),
            (b00, +1), (b01, -1),
            (c00, +1)
        ])
        # M7 = (A01-A00)*(B10+B11)
        entries.append([
            (a01, +1), (a00, -1),
            (b10, +1), (b11, +1),
            (c11, +1)
        ])
        
        return entries
    
    def apply_strassen_block(a_row_idx, a_col_start, b_row_start, b_col_start, c_row_idx, c_col_start, col_offset):
        """Apply one 2x2x2 Strassen decomposition."""
        a_rows = [a_row_idx, a_row_idx + 1]  # Wait, no. a_row_idx should be the two row indices.
        # Actually, for 2x4x4, the A submatrix is 2 rows x 2 cols
        # Let me restructure: pass the actual row/col indices
        pass
    
    # Let me redo this more carefully.
    # For the 2x4x4 part:
    # A rows: 0, 1 (indices in U: 0*m+0=0, 0*m+1=1, 1*m+0=4, 1*m+1=5, 1*m+2=6, 1*m+3=7)
    # Wait, U is indexed by [i*m + j] where i is row of A, j is col of A.
    # A is 3x4, so U has 12 rows: row i, col j -> index i*4+j
    # V is indexed by [j*p + k] where j is row of B, k is col of B.
    # B is 4x4, so V has 16 rows: row j, col k -> index j*4+k
    # W is indexed by [k*n + i] where k is col of output (=row of B), i is row of output.
    # Output C is 3x4, so W has 12 rows: output row i, col k -> index k*3+i
    
    # For 2x4x4: A rows 0,1; A cols 0,1,2,3; B rows 0,1,2,3; B cols 0,1,2,3; output rows 0,1; output cols 0,1,2,3
    
    # Split into 4 quadrants of B:
    # B_TL: rows 0,1 cols 0,1
    # B_TR: rows 0,1 cols 2,3  
    # B_BL: rows 2,3 cols 0,1
    # B_BR: rows 2,3 cols 2,3
    
    # A_L: cols 0,1
    # A_R: cols 2,3
    
    # C_L (output cols 0,1) = A_L * B_TL + A_R * B_BL
    # C_R (output cols 2,3) = A_L * B_TR + A_R * B_BR
    
    # Each term is a 2x2 x 2x2 -> 2x2 product, decomposable with Strassen (7 terms)
    
    def add_strassen_term(A_row0, A_row1, A_col0, A_col1, B_row0, B_row1, B_col0, B_col1, C_row0, C_row1, C_col0, C_col1, col):
        """Add one Strassen 2x2x2 decomposition at column `col`."""
        # A indices in U: A_row * m + A_col
        # B indices in V: B_row * p + B_col  
        # C indices in W: C_col * n + C_row
        
        a00_u = A_row0 * m + A_col0
        a01_u = A_row0 * m + A_col1
        a10_u = A_row1 * m + A_col0
        a11_u = A_row1 * m + A_col1
        
        b00_v = B_row0 * p + B_col0
        b01_v = B_row0 * p + B_col1
        b10_v = B_row1 * p + B_col0
        b11_v = B_row1 * p + B_col1
        
        c00_w = C_col0 * n + C_row0
        c01_w = C_col1 * n + C_row0
        c10_w = C_col0 * n + C_row1
        c11_w = C_col1 * n + C_row1
        
        # M1 = (A00-A11)*(B11-B00) -> C00 += M1, C11 += M1
        U[a00_u][col] += 1
        U[a11_u][col] += -1
        V[b11_v][col] += 1
        V[b00_v][col] += -1
        W[c00_w][col] += 1
        W[c11_w][col] += 1
        
        col += 1
        # M2 = (A00+A01)*B11 -> C00 -= M2, C01 += M2
        U[a00_u][col] += 1
        U[a01_u][col] += 1
        V[b11_v][col] += 1
        W[c00_w][col] += -1
        W[c01_w][col] += 1
        
        col += 1
        # M3 = A00*(B00-B11) -> C01 += M3, C11 -= M3
        U[a00_u][col] += 1
        V[b00_v][col] += 1
        V[b11_v][col] += -1
        W[c01_w][col] += 1
        W[c11_w][col] += -1
        
        col += 1
        # M4 = A11*(B00+B11) -> C00 += M4, C10 += M4
        U[a11_u][col] += 1
        V[b00_v][col] += 1
        V[b11_v][col] += 1
        W[c00_w][col] += 1
        W[c10_w][col] += 1
        
        col += 1
        # M5 = (A01-A11)*(B00+B10) -> C10 += M5, C11 += M5
        U[a01_u][col] += 1
        U[a11_u][col] += -1
        V[b00_v][col] += 1
        V[b10_v][col] += 1
        W[c10_w][col] += 1
        W[c11_w][col] += 1
        
        col += 1
        # M6 = (A00+A11)*(B00-B01) -> C00 += M6
        U[a00_u][col] += 1
        U[a11_u][col] += 1
        V[b00_v][col] += 1
        V[b01_v][col] += -1
        W[c00_w][col] += 1
        
        col += 1
        # M7 = (A01-A00)*(B10+B11) -> C11 += M7
        U[a01_u][col] += 1
        U[a00_u][col] += -1
        V[b10_v][col] += 1
        V[b11_v][col] += 1
        W[c11_w][col] += 1
        
        col += 1
        return col
    
    # C_L = A_L * B_TL + A_R * B_BL
    # Term 1: A_L (rows 0,1; cols 0,1) * B_TL (rows 0,1; cols 0,1) -> C rows 0,1; cols 0,1
    col = add_strassen_term(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, col)
    # Term 2: A_R (rows 0,1; cols 2,3) * B_BL (rows 2,3; cols 0,1) -> C rows 0,1; cols 0,1
    col = add_strassen_term(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, col)
    
    # C_R = A_L * B_TR + A_R * B_BR
    # Term 3: A_L (rows 0,1; cols 0,1) * B_TR (rows 0,1; cols 2,3) -> C rows 0,1; cols 2,3
    col = add_strassen_term(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, col)
    # Term 4: A_R (rows 0,1; cols 2,3) * B_BR (rows 2,3; cols 2,3) -> C rows 0,1; cols 2,3
    col = add_strassen_term(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, col)
    
    # col should be 16 + 28 = 44
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
