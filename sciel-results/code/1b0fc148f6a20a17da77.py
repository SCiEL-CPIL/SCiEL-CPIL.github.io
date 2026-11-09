import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a rank-1 term
    # U_indices: list of (row, col, coeff) in A (n x m)
    # V_indices: list of (row, col, coeff) in B (m x p)
    # W_indices: list of (row, col, coeff) in C (p x n)
    # col: the rank index to fill
    
    def add_term(col, u_terms, v_terms, w_terms):
        for r, c, val in u_terms:
            U[r * m + c, col] += val
        for r, c, val in v_terms:
            V[r * p + c, col] += val
        for r, c, val in w_terms:
            W[r * n + c, col] += val
            
    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """
        Adds Strassen decomposition for A[a_row_off:a_row_off+2, a_col_off:a_col_off+2] * 
        B[b_row_off:b_row_off+2, b_col_off:b_col_off+2] -> 
        C[c_row_off:c_row_off+2, c_col_off:c_col_off+2]
        """
        # A indices relative to block
        # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
        # B indices relative to block
        # B00: (0,0), B01: (0,1), B10: (1,0), B11: (1,1)
        # C indices relative to block
        # C00: (0,0), C01: (0,1), C10: (1,0), C11: (1,1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        u1 = [(a_row_off + 0, a_col_off + 0, 1), (a_row_off + 1, a_col_off + 1, 1)]
        v1 = [(b_row_off + 0, b_col_off + 0, 1), (b_row_off + 1, b_col_off + 1, 1)]
        w1 = [(c_row_off + 0, c_col_off + 0, 1), (c_row_off + 1, c_col_off + 1, 1)] # C00, C11
        add_term(col, u1, v1, w1)
        col += 1
        
        # M2 = (A10 + A11) * B00
        u2 = [(a_row_off + 1, a_col_off + 0, 1), (a_row_off + 1, a_col_off + 1, 1)]
        v2 = [(b_row_off + 0, b_col_off + 0, 1)]
        w2 = [(c_row_off + 1, c_col_off + 0, 1), (c_row_off + 1, c_col_off + 1, -1)] # C10, C11
        add_term(col, u2, v2, w2)
        col += 1
        
        # M3 = A00 * (B10 - B11)
        u3 = [(a_row_off + 0, a_col_off + 0, 1)]
        v3 = [(b_row_off + 1, b_col_off + 0, 1), (b_row_off + 1, b_col_off + 1, -1)]
        w3 = [(c_row_off + 1, c_col_off + 1, 1)] # C11
        add_term(col, u3, v3, w3)
        col += 1
        
        # M4 = A11 * (B01 + B11)
        u4 = [(a_row_off + 1, a_col_off + 1, 1)]
        v4 = [(b_row_off + 0, b_col_off + 1, 1), (b_row_off + 1, b_col_off + 1, 1)]
        w4 = [(c_row_off + 0, c_col_off + 0, 1), (c_row_off + 1, c_col_off + 0, 1)] # C00, C10
        add_term(col, u4, v4, w4)
        col += 1
        
        # M5 = (A00 + A01) * B11
        u5 = [(a_row_off + 0, a_col_off + 0, 1), (a_row_off + 0, a_col_off + 1, 1)]
        v5 = [(b_row_off + 1, b_col_off + 1, 1)]
        w5 = [(c_row_off + 0, c_col_off + 0, -1), (c_row_off + 0, c_col_off + 1, 1)] # C00, C01
        add_term(col, u5, v5, w5)
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        u6 = [(a_row_off + 1, a_col_off + 0, 1), (a_row_off + 0, a_col_off + 0, -1)]
        v6 = [(b_row_off + 0, b_col_off + 0, 1), (b_row_off + 0, b_col_off + 1, 1)]
        w6 = [(c_row_off + 1, c_col_off + 1, 1)] # C11
        add_term(col, u6, v6, w6)
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        u7 = [(a_row_off + 0, a_col_off + 1, 1), (a_row_off + 1, a_col_off + 1, -1)]
        v7 = [(b_row_off + 1, b_col_off + 0, 1), (b_row_off + 1, b_col_off + 1, 1)]
        w7 = [(c_row_off + 0, c_col_off + 0, -1), (c_row_off + 0, c_col_off + 1, 1)] # C00, C01
        add_term(col, u7, v7, w7)
        col += 1
        
        return col

    def add_schoolbook_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """
        Adds schoolbook decomposition for A[2x2] * B[2x1] -> C[2x1]
        A: (2x2), B: (2x1), C: (2x1)
        B_col_off is the column in B (size 1 wide)
        C_col_off is the column in C (size 1 wide)
        """
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10
        
        # Term 1: A00 * B00 -> C0
        u1 = [(a_row_off + 0, a_col_off + 0, 1)]
        v1 = [(b_row_off + 0, b_col_off + 0, 1)]
        w1 = [(c_row_off + 0, c_col_off + 0, 1)]
        add_term(col, u1, v1, w1)
        col += 1
        
        # Term 2: A01 * B10 -> C0
        u2 = [(a_row_off + 0, a_col_off + 1, 1)]
        v2 = [(b_row_off + 1, b_col_off + 0, 1)]
        w2 = [(c_row_off + 0, c_col_off + 0, 1)]
        add_term(col, u2, v2, w2)
        col += 1
        
        # Term 3: A10 * B00 -> C1
        u3 = [(a_row_off + 1, a_col_off + 0, 1)]
        v3 = [(b_row_off + 0, b_col_off + 0, 1)]
        w3 = [(c_row_off + 1, c_col_off + 0, 1)]
        add_term(col, u3, v3, w3)
        col += 1
        
        # Term 4: A11 * B10 -> C1
        u4 = [(a_row_off + 1, a_col_off + 1, 1)]
        v4 = [(b_row_off + 1, b_col_off + 0, 1)]
        w4 = [(c_row_off + 1, c_col_off + 0, 1)]
        add_term(col, u4, v4, w4)
        col += 1
        
        return col

    col = 0
    
    # Block 1: A_L (rows 0-1, cols 0-1) * B_T1 (rows 0-1, cols 0-1) -> C (rows 0-1, cols 0-1)
    # A_L is A[0:2, 0:2]
    # B_T1 is B[0:2, 0:2]
    # C is C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A_L (rows 0-1, cols 0-1) * B_T2 (rows 0-1, cols 2-3) -> C (rows 0-1, cols 2-3)
    # A_L is A[0:2, 0:2]
    # B_T2 is B[0:2, 2:4]
    # C is C[0:2, 2:4]
    col = add_strassen_block(0, 0, 0, 2, 2, 0)
    
    # Block 3: A_L (rows 0-1, cols 0-1) * B_T3 (rows 0-1, col 4) -> C (rows 0-1, col 4)
    # A_L is A[0:2, 0:2]
    # B_T3 is B[0:2, 4:5]
    # C is C[0:2, 4:5]
    col = add_schoolbook_block(0, 0, 0, 4, 4, 0)
    
    # Block 4: A_R (rows 0-1, cols 2-3) * B_B1 (rows 2-3, cols 0-1) -> C (rows 0-1, cols 0-1)
    # A_R is A[0:2, 2:4]
    # B_B1 is B[2:4, 0:2]
    # C is C[0:2, 0:2]
    col = add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 5: A_R (rows 0-1, cols 2-3) * B_B2 (rows 2-3, cols 2-3) -> C (rows 0-1, cols 2-3)
    # A_R is A[0:2, 2:4]
    # B_B2 is B[2:4, 2:4]
    # C is C[0:2, 2:4]
    col = add_strassen_block(0, 2, 2, 2, 2, 0)
    
    # Block 6: A_R (rows 0-1, cols 2-3) * B_B3 (rows 2-3, col 4) -> C (rows 0-1, col 4)
    # A_R is A[0:2, 2:4]
    # B_B3 is B[2:4, 4:5]
    # C is C[0:2, 4:5]
    col = add_schoolbook_block(0, 2, 2, 4, 4, 0)
    
    # Convert to lists
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
