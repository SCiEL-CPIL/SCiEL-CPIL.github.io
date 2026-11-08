"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""

def construct():
    n, m, p = 2, 4, 5
    
    # We'll use a block decomposition approach:
    # Split the 4 columns of B into two 2-column blocks.
    # Block 1: A * B_1 where B_1 is 4x2 -> C_1 is 2x2
    # Block 2: A * B_2 where B_2 is 4x2 -> C_2 is 2x2
    # Block 3: A * B_3 where B_3 is 4x1 -> C_3 is 2x1
    
    # For each 2x4x2 sub-problem (A is 2x4, B_sub is 4x2, C_sub is 2x2):
    # Split A into A_top (2x2, cols 0,1) and A_bottom (2x2, cols 2,3)
    # Split B_sub into B_top (2x2, rows 0,1) and B_bottom (2x2, rows 2,3)
    # C_sub = A_top * B_top + A_bottom * B_bottom
    # Each 2x2x2 multiplication uses Strassen (rank 7)
    # So each 2x4x2 has rank at most 14
    
    # For 2x4x1 (A is 2x4, B is 4x1, C is 2x1):
    # Split A into A_top (2x2) and A_bottom (2x2)
    # Split B into B_top (2x1) and B_bottom (2x1)
    # C = A_top * B_top + A_bottom * B_bottom
    # Each 2x2x1 has rank 4 (standard)
    # So 2x4x1 has rank 8
    
    # Total rank: 14 + 14 + 8 = 36
    
    # Strassen 2x2x2 decomposition:
    # For A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition, each of shape (4,7)."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1  # A11 (i=0,j=0)
        U[3][0] = 1  # A22 (i=1,j=1)
        V[0][0] = 1  # B11 (j=0,k=0)
        V[3][0] = 1  # B22 (j=1,k=1)
        W[0][0] = 1  # C11 (k=0,i=0)
        W[3][0] = 1  # C22 (k=1,i=1)
        
        # M2 = (A21+A22)B11
        U[2][1] = 1  # A21 (i=1,j=0)
        U[3][1] = 1  # A22 (i=1,j=1)
        V[0][1] = 1  # B11 (j=0,k=0)
        W[2][1] = 1  # C21 (k=1,i=0)
        
        # M3 = A11(B21-B22)
        U[0][2] = 1  # A11 (i=0,j=0)
        V[2][2] = 1  # B21 (j=1,k=0)
        V[3][2] = -1 # B22 (j=1,k=1)
        W[1][2] = 1  # C12 (k=0,i=1)
        W[3][2] = 1  # C22 (k=1,i=1)
        
        # M4 = A22(B12-B11)
        U[3][3] = 1  # A22 (i=1,j=1)
        V[1][3] = 1  # B12 (j=0,k=1)
        V[0][3] = -1 # B11 (j=0,k=0)
        W[1][3] = 1  # C12 (k=0,i=1)
        W[2][3] = 1  # C21 (k=1,i=0)
        
        # M5 = (A11+A12)B22
        U[0][4] = 1  # A11 (i=0,j=0)
        U[1][4] = 1  # A12 (i=0,j=1)
        V[3][4] = 1  # B22 (j=1,k=1)
        W[0][4] = -1 # C11 (k=0,i=0)
        W[1][4] = 1  # C12 (k=0,i=1)
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1  # A21 (i=1,j=0)
        U[0][5] = -1 # A11 (i=0,j=0)
        V[0][5] = 1  # B11 (j=0,k=0)
        V[1][5] = 1  # B12 (j=0,k=1)
        W[3][5] = 1  # C22 (k=1,i=1)
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1  # A12 (i=0,j=1)
        U[3][6] = -1 # A22 (i=1,j=1)
        V[2][6] = 1  # B21 (j=1,k=0)
        V[3][6] = 1  # B22 (j=1,k=1)
        W[0][6] = 1  # C11 (k=0,i=0)
        W[1][6] = 1  # C12 (k=0,i=1)
        
        return U, V, W
    
    def standard_2x2x1():
        """Return (U, V, W) for 2x2x1 standard decomposition, each of shape (4,4), (2,4), (2,4)."""
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # For 2x2x1: n=2, m=2, p=1
        # A indices: i*2+j, i in {0,1}, j in {0,1} -> 0,1,2,3
        # B indices: j*1+0 = j, j in {0,1} -> 0,1
        # C indices: 0*2+i = i, i in {0,1} -> 0,1
        
        col = 0
        for i in range(2):
            for j in range(2):
                U[i*2+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        
        return U, V, W
    
    sU, sV, sW = strassen_2x2x2()
    stdU, stdV, stdW = standard_2x2x1()
    
    n_full, m_full, p_full = 2, 4, 5
    rank_total = 36
    U_full = [[0]*rank_total for _ in range(n_full*m_full)]  # 8 x 36
    V_full = [[0]*rank_total for _ in range(m_full*p_full)]  # 20 x 36
    W_full = [[0]*rank_total for _ in range(p_full*n_full)]  # 10 x 36
    
    col_offset = 0
    
    # Helper function to add a Strassen block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add a Strassen 2x2x2 block.
        a_row_offset: offset in A rows (0 or 1, but here A has only 2 rows so always 0)
        a_col_offset: offset in A columns (0 or 2)
        b_row_offset: offset in B rows (0 or 2)
        b_col_offset: offset in B columns (0, 2, or 4)
        c_row_offset: offset in C columns (0, 2, or 4) - note: C is indexed by k*n+i
        c_col_offset: not used, C row offset determines the column in C
        """
        nonlocal col_offset
        for c in range(7):
            for la in range(4):
                # la = i*2+j for local 2x2
                i = la // 2
                j = la % 2
                ga = (a_row_offset + i) * 4 + (a_col_offset + j)
                U_full[ga][col_offset + c] = sU[la][c]
            for lb in range(4):
                # lb = j*2+k for local 2x2
                j = lb // 2
                k = lb % 2
                gb = (b_row_offset + j) * 5 + (b_col_offset + k)
                V_full[gb][col_offset + c] = sV[lb][c]
            for lc in range(4):
                # lc = k*2+i for local 2x2, C index is k*n+i
                k = lc // 2
                i = lc % 2
                gc = (c_row_offset + k) * 2 + i
                W_full[gc][col_offset + c] = sW[lc][c]
        col_offset += 7
    
    # Helper function to add a standard 2x2x1 block
    def add_standard_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        """
        Add a standard 2x2x1 block.
        a_row_offset: offset in A rows
        a_col_offset: offset in A columns (0 or 2)
        b_row_offset: offset in B rows (0 or 2)
        b_col_offset: offset in B columns (4 for the last column)
        c_col_offset: offset in C columns
        """
        nonlocal col_offset
        for c in range(4):
            for la in range(4):
                i = la // 2
                j = la % 2
                ga = (a_row_offset + i) * 4 + (a_col_offset + j)
                U_full[ga][col_offset + c] = stdU[la][c]
            for lb in range(2):
                j = lb
                gb = (b_row_offset + j) * 5 + b_col_offset
                V_full[gb][col_offset + c] = stdV[lb][c]
            for lc in range(2):
                i = lc
                gc = c_col_offset * 2 + i
                W_full[gc][col_offset + c] = stdW[lc][c]
        col_offset += 4
    
    # Block 1a: A_top (cols 0,1) * B_top (rows 0,1, cols 0,1)
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 1b: A_bottom (cols 2,3) * B_bottom (rows 2,3, cols 0,1)
    add_strassen_block(0, 2, 2, 0, 0)
    
    # Block 2a: A_top (cols 0,1) * B_top (rows 0,1, cols 2,3)
    add_strassen_block(0, 0, 0, 2, 2)
    
    # Block 2b: A_bottom (cols 2,3) * B_bottom (rows 2,3, cols 2,3)
    add_strassen_block(0, 2, 2, 2, 2)
    
    # Block 3a: A_top (cols 0,1) * B_top (rows 0,1, col 4)
    add_standard_block(0, 0, 0, 4, 4)
    
    # Block 3b: A_bottom (cols 2,3) * B_bottom (rows 2,3, col 4)
    add_standard_block(0, 2, 2, 4, 4)
    
    assert col_offset == rank_total, f"Expected col_offset={rank_total}, got {col_offset}"
    
    return {"U": U_full, "V": V_full, "W": W_full}
