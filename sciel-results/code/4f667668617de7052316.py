import numpy as np

def construct():
    """
    Construct a rank-36 exact tensor decomposition for 2x4x5 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and schoolbook for 2x2x1 blocks.
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        for (i, j), val in u_coeffs.items():
            U[i*m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j*p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k*n + i, idx] += val

    def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
        # a_row_offset: row in A (0 for A_L, 1 for A_R in block coords? No, 0-1 in 2x2 block)
        # b_row_offset: row in B (0 for B_T, 1 for B_B)
        # c_col_offset: col in C (0 or 1 within the 2-col block)
        
        # A is 2x2, B is 2x2, C is 2x2
        # A_L[i,j] is at A[i, a_row_offset*2 + j]
        # B_T[k,l] is at B[b_row_offset*2 + k, l] -- wait, b_row_offset is row index in B (0..3)
        # In our block decomposition:
        # A_L = A[0:2, 0:2]
        # A_R = A[0:2, 2:4]
        # B_T = B[0:2, :]
        # B_B = B[2:4, :]
        
        # The function args are offsets into the 2x2 local coordinate system
        # Actually, let's make it take global offsets.
        
        pass

    # We will manually construct the terms for the specific blocks.
    
    # Block 1: A_L (cols 0,1 of A) * B_T (rows 0,1 of B)
    # A_L is 2x2. B_T11 is B[0:2, 0:2]. 
    idx = 0
    a_off = 0 # A row 0? No, A is 2x4. We split cols.
    # A_L corresponds to cols 0,1.
    # A_R corresponds to cols 2,3.
    
    def strassen_2x2x2(a_col_start, b_row_start, c_col_start, idx_start):
        # a_col_start: 0 or 2 (offset in A's 4 cols)
        # b_row_start: 0 or 2 (offset in B's 4 rows)
        # c_col_start: 0, 2, or 4 (offset in C's 5 cols, must be 0 or 2 for 2x2)
        
        # M1
        add_term(idx_start, 
                 {(0, a_col_start): 1, (1, a_col_start+1): 1},
                 {(b_row_start, c_col_start): 1, (b_row_start+1, c_col_start+1): 1},
                 {(c_col_start, 0): 1, (c_col_start+1, 1): 1})
        # M2
        add_term(idx_start+1,
                 {(1, a_col_start): 1, (1, a_col_start+1): 1},
                 {(b_row_start, c_col_start): 1},
                 {(c_col_start, 1): 1, (c_col_start+1, 1): -1})
        # M3
        add_term(idx_start+2,
                 {(0, a_col_start): 1},
                 {(b_row_start, c_col_start+1): 1, (b_row_start+1, c_col_start+1): -1},
                 {(c_col_start+1, 0): 1, (c_col_start+1, 1): 1})
        # M4
        add_term(idx_start+3,
                 {(1, a_col_start+1): 1},
                 {(b_row_start+1, c_col_start): 1, (b_row_start, c_col_start): -1},
                 {(c_col_start, 0): 1, (c_col_start, 1): 1})
        # M5
        add_term(idx_start+4,
                 {(0, a_col_start): 1, (0, a_col_start+1): 1},
                 {(b_row_start+1, c_col_start+1): 1},
                 {(c_col_start, 0): -1, (c_col_start+1, 0): 1})
        # M6
        add_term(idx_start+5,
                 {(1, a_col_start): 1, (0, a_col_start): -1},
                 {(b_row_start, c_col_start): 1, (b_row_start, c_col_start+1): 1},
                 {(c_col_start+1, 1): 1})
        # M7
        add_term(idx_start+6,
                 {(0, a_col_start+1): 1, (1, a_col_start+1): -1},
                 {(b_row_start+1, c_col_start): 1, (b_row_start+1, c_col_start+1): 1},
                 {(c_col_start, 0): 1})
        
    def schoolbook_2x2x1(a_col_start, b_row_start, c_col_start, idx_start):
        # a_col_start: 0 or 2
        # b_row_start: 0 or 2
        # c_col_start: 4 (only 1 col)
        
        # Term 1: A00 B0 -> C0
        add_term(idx_start,
                 {(0, a_col_start): 1},
                 {(b_row_start, c_col_start): 1},
                 {(c_col_start, 0): 1})
        # Term 2: A01 B1 -> C0
        add_term(idx_start+1,
                 {(0, a_col_start+1): 1},
                 {(b_row_start+1, c_col_start): 1},
                 {(c_col_start, 0): 1})
        # Term 3: A10 B0 -> C1
        add_term(idx_start+2,
                 {(1, a_col_start): 1},
                 {(b_row_start, c_col_start): 1},
                 {(c_col_start, 1): 1})
        # Term 4: A11 B1 -> C1
        add_term(idx_start+3,
                 {(1, a_col_start+1): 1},
                 {(b_row_start+1, c_col_start): 1},
                 {(c_col_start, 1): 1})

    # Cols 0,1 of C
    # From A_L * B_T[0:2] (B rows 0,1)
    strassen_2x2x2(0, 0, 0, idx); idx += 7
    # From A_R * B_B[0:2] (B rows 2,3)
    strassen_2x2x2(2, 2, 0, idx); idx += 7
    
    # Cols 2,3 of C
    # From A_L * B_T[2:4] (B rows 0,1)
    strassen_2x2x2(0, 0, 2, idx); idx += 7
    # From A_R * B_B[2:4] (B rows 2,3)
    strassen_2x2x2(2, 2, 2, idx); idx += 7
    
    # Col 4 of C
    # From A_L * B_T[4:5] (B rows 0,1)
    schoolbook_2x2x1(0, 0, 4, idx); idx += 4
    # From A_R * B_B[4:5] (B rows 2,3)
    schoolbook_2x2x1(2, 2, 4, idx); idx += 4
    
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
