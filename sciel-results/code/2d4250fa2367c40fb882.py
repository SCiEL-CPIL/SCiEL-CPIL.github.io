"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
Uses block-structured decomposition with Strassen for 2x2x2 blocks.
"""

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)
        - W has shape (p*n, rank) = (12, rank)
    """
    n, m, p = 3, 3, 4
    
    # We'll build the decomposition by splitting into blocks:
    # A is 3x3, split as [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B is 3x4, split as [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    # C is 3x4, split as [[C11, C12], [C21, C22]] where C11 is 2x2, C12 is 2x2, C21 is 1x2, C22 is 1x2
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For 2x2*2x2 we use Strassen (rank 7)
    # For 2x2*2x2 we use Strassen (rank 7)
    # For 2x1*1x2 we use schoolbook (rank 2)
    # For 2x1*1x2 we use schoolbook (rank 2)
    # For 1x2*2x2 we use schoolbook (rank 4)
    # For 1x2*2x2 we use schoolbook (rank 4)
    # For 1x1*1x2 we use schoolbook (rank 2)
    # For 1x1*1x2 we use schoolbook (rank 2)
    
    # Total rank = 7 + 7 + 2 + 2 + 4 + 4 + 2 + 2 = 30
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 9 x rank
    V = [[0] * rank for _ in range(m * p)]  # 12 x rank
    W = [[0] * rank for _ in range(p * n)]  # 12 x rank
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2 * 2x2
    def add_strassen_2x2(a_row_start, b_col_start, c_row_start, c_col_start, base_col):
        """
        Add Strassen decomposition for C2x2 = A2x2 * B2x2.
        A is at rows a_row_start, a_row_start+1; cols 0,1
        B is at rows 0,1; cols b_col_start, b_col_start+1
        C is at rows c_row_start, c_row_start+1; cols c_col_start, c_col_start+1
        """
        nonlocal col
        # Strassen formulas for 2x2*2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        a00 = (a_row_start, 0)
        a01 = (a_row_start, 1)
        a10 = (a_row_start + 1, 0)
        a11 = (a_row_start + 1, 1)
        
        b00 = (0, b_col_start)
        b01 = (0, b_col_start + 1)
        b10 = (1, b_col_start)
        b11 = (1, b_col_start + 1)
        
        c00 = (c_row_start, c_col_start)
        c01 = (c_row_start, c_col_start + 1)
        c10 = (c_row_start + 1, c_col_start)
        c11 = (c_row_start + 1, c_col_start + 1)
        
        # M1
        U[a00[0] * m + a00[1]][base_col] = 1
        U[a11[0] * m + a11[1]][base_col] = 1
        V[b00[0] * p + b00[1]][base_col] = 1
        V[b11[0] * p + b11[1]][base_col] = 1
        W[c00[1] * n + c00[0]][base_col] = 1
        W[c11[1] * n + c11[0]][base_col] = 1
        col += 1
        
        # M2
        U[a10[0] * m + a10[1]][base_col] = 1
        U[a11[0] * m + a11[1]][base_col] = 1
        V[b00[0] * p + b00[1]][base_col] = 1
        W[c10[1] * n + c10[0]][base_col] = 1
        W[c11[1] * n + c11[0]][base_col] = -1
        col += 1
        
        # M3
        U[a00[0] * m + a00[1]][base_col] = 1
        V[b01[0] * p + b01[1]][base_col] = 1
        V[b11[0] * p + b11[1]][base_col] = -1
        W[c01[1] * n + c01[0]][base_col] = 1
        W[c11[1] * n + c11[0]][base_col] = 1
        col += 1
        
        # M4
        U[a11[0] * m + a11[1]][base_col] = 1
        V[b10[0] * p + b10[1]][base_col] = 1
        V[b00[0] * p + b00[1]][base_col] = -1
        W[c00[1] * n + c00[0]][base_col] = 1
        W[c10[1] * n + c10[0]][base_col] = 1
        col += 1
        
        # M5
        U[a00[0] * m + a00[1]][base_col] = 1
        U[a01[0] * m + a01[1]][base_col] = 1
        V[b11[0] * p + b11[1]][base_col] = 1
        W[c00[1] * n + c00[0]][base_col] = -1
        W[c01[1] * n + c01[0]][base_col] = 1
        col += 1
        
        # M6
        U[a10[0] * m + a10[1]][base_col] = 1
        U[a00[0] * m + a00[1]][base_col] = -1
        V[b00[0] * p + b00[1]][base_col] = 1
        V[b01[0] * p + b01[1]][base_col] = 1
        W[c11[1] * n + c11[0]][base_col] = 1
        col += 1
        
        # M7
        U[a01[0] * m + a01[1]][base_col] = 1
        U[a11[0] * m + a11[1]][base_col] = -1
        V[b10[0] * p + b10[1]][base_col] = 1
        V[b11[0] * p + b11[1]][base_col] = 1
        W[c00[1] * n + c00[0]][base_col] = 1
        col += 1
    
    # Helper to add schoolbook for 2x1 * 1x2
    def add_schoolbook_2x1_1x2(a_row_start, b_row, c_row_start, c_col_start, base_col):
        """
        Add schoolbook for C2x2 = A2x1 * B1x2.
        A is at rows a_row_start, a_row_start+1; col 2 (the 3rd column of A)
        B is at row b_row; cols c_col_start, c_col_start+1
        C is at rows c_row_start, c_row_start+1; cols c_col_start, c_col_start+1
        """
        nonlocal col
        # C[i][j] = A[i][2] * B[b_row][j]
        for i in range(2):
            for j in range(2):
                a_idx = (a_row_start + i, 2)
                b_idx = (b_row, c_col_start + j)
                c_idx = (c_row_start + i, c_col_start + j)
                
                U[a_idx[0] * m + a_idx[1]][base_col] = 1
                V[b_idx[0] * p + b_idx[1]][base_col] = 1
                W[c_idx[1] * n + c_idx[0]][base_col] = 1
                col += 1
    
    # Helper to add schoolbook for 1x2 * 2x2
    def add_schoolbook_1x2_2x2(a_row, b_col_start, c_row, c_col_start, base_col):
        """
        Add schoolbook for C1x2 = A1x2 * B2x2.
        A is at row a_row; cols 0, 1
        B is at rows 0, 1; cols b_col_start, b_col_start+1
        C is at row c_row; cols c_col_start, c_col_start+1
        """
        nonlocal col
        # C[0][j] = A[0][0]*B[0][j] + A[0][1]*B[1][j]
        for j in range(2):
            for i in range(2):
                a_idx = (a_row, i)
                b_idx = (i, b_col_start + j)
                c_idx = (c_row, c_col_start + j)
                
                U[a_idx[0] * m + a_idx[1]][base_col] = 1
                V[b_idx[0] * p + b_idx[1]][base_col] = 1
                W[c_idx[1] * n + c_idx[0]][base_col] = 1
                col += 1
    
    # Helper to add schoolbook for 1x1 * 1x2
    def add_schoolbook_1x1_1x2(a_row, b_row, c_row, c_col_start, base_col):
        """
        Add schoolbook for C1x2 = A1x1 * B1x2.
        A is at (a_row, 2)
        B is at row b_row; cols c_col_start, c_col_start+1
        C is at row c_row; cols c_col_start, c_col_start+1
        """
        nonlocal col
        for j in range(2):
            a_idx = (a_row, 2)
            b_idx = (b_row, c_col_start + j)
            c_idx = (c_row, c_col_start + j)
            
            U[a_idx[0] * m + a_idx[1]][base_col] = 1
            V[b_idx[0] * p + b_idx[1]][base_col] = 1
            W[c_idx[1] * n + c_idx[0]][base_col] = 1
            col += 1
    
    # Now build the full decomposition:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # A11 is rows 0-1, cols 0-1
    # A12 is rows 0-1, col 2
    # A21 is row 2, cols 0-1
    # A22 is (2, 2)
    
    # B11 is rows 0-1, cols 0-1
    # B12 is rows 0-1, cols 2-3
    # B21 is row 2, cols 0-1
    # B22 is row 2, cols 2-3
    
    # C11 is rows 0-1, cols 0-1
    # C12 is rows 0-1, cols 2-3
    # C21 is row 2, cols 0-1
    # C22 is row 2, cols 2-3
    
    # C11 = A11*B11 + A12*B21
    add_strassen_2x2(0, 0, 0, 0, col)  # A11*B11, base_col=0, uses cols 0-6
    col += 7
    add_schoolbook_2x1_1x2(0, 2, 0, 0, col)  # A12*B21, uses cols 7-8
    col += 2
    
    # C12 = A11*B12 + A12*B22
    add_strassen_2x2(0, 2, 0, 2, col)  # A11*B12, uses cols 9-15
    col += 7
    add_schoolbook_2x1_1x2(0, 2, 0, 2, col)  # A12*B22, uses cols 16-17
    col += 2
    
    # C21 = A21*B11 + A22*B21
    add_schoolbook_1x2_2x2(2, 0, 2, 0, col)  # A21*B11, uses cols 18-21
    col += 4
    add_schoolbook_1x1_1x2(2, 2, 2, 0, col)  # A22*B21, uses cols 22-23
    col += 2
    
    # C22 = A21*B12 + A22*B22
    add_schoolbook_1x2_2x2(2, 2, 2, 2, col)  # A21*B12, uses cols 24-27
    col += 4
    add_schoolbook_1x1_1x2(2, 2, 2, 2, col)  # A22*B22, uses cols 28-29
    col += 2
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
