import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    # We will build the decomposition by partitioning the 3x4x7 tensor into:
    # - Two 2x2x7 blocks (rows 0-1 of A, cols 0-1 and 2-3 of A)
    # - Two 1x2x7 blocks (row 2 of A, cols 0-1 and 2-3 of A)
    
    # For each 2x2x7 block, we use:
    #   - Three Strassen 2x2x2 decompositions (for cols 0-1, 2-3, 4-5 of B)
    #   - One standard 2x2x1 decomposition (for col 6 of B)
    # Total rank per 2x2x7 block: 3*7 + 4 = 25
    # 
    # For each 1x2x7 block, we use:
    #   - Seven standard 1x2x1 decompositions (rank 2 each)
    # Total rank per 1x2x7 block: 7*2 = 14
    #
    # Total rank: 2*25 + 2*14 = 50 + 28 = 78
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]  # 12 x 78
    V = [[0] * rank for _ in range(m * p)]  # 28 x 78
    W = [[0] * rank for _ in range(p * n)]  # 21 x 78
    
    rank_offset = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank_offset
        U[u_idx][rank_offset] += u_val
        V[v_idx][rank_offset] += v_val
        W[w_idx][rank_offset] += w_val
        rank_offset += 1
    
    def strassen_2x2x2(row_offset, col_offset, b_col_offset):
        """
        Add a Strassen decomposition for A[row_offset:row_offset+2, col_offset:col_offset+2]
        * B[col_offset:col_offset+2, b_col_offset:b_col_offset+2]
        -> C[b_col_offset:b_col_offset+2, row_offset:row_offset+2]
        
        Strassen's 7 products:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B10)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # A indices: A[i, j] -> U[i*m + j]
        # B indices: B[j, k] -> V[j*p + k]
        # C indices: C[k, i] -> W[k*n + i]
        
        a00 = row_offset * m + col_offset
        a01 = row_offset * m + col_offset + 1
        a10 = (row_offset + 1) * m + col_offset
        a11 = (row_offset + 1) * m + col_offset + 1
        
        b00 = col_offset * p + b_col_offset
        b01 = col_offset * p + b_col_offset + 1
        b10 = (col_offset + 1) * p + b_col_offset
        b11 = (col_offset + 1) * p + b_col_offset + 1
        
        c00 = b_col_offset * n + row_offset
        c01 = (b_col_offset + 1) * n + row_offset
        c10 = b_col_offset * n + row_offset + 1
        c11 = (b_col_offset + 1) * n + row_offset + 1
        
        # M1 = (A00+A11)(B00+B11)
        add_term(a00, b00, c00, 1, 1, 1)
        add_term(a00, b11, c00, 1, 1, 1)
        add_term(a11, b00, c00, 1, 1, 1)
        add_term(a11, b11, c00, 1, 1, 1)
        
        # M2 = (A10+A11)B00
        add_term(a10, b00, c10, 1, 1, 1)
        add_term(a11, b00, c10, 1, 1, 1)
        
        # M3 = A00(B10-B11)
        add_term(a00, b10, c01, 1, 1, 1)
        add_term(a00, b11, c01, 1, -1, 1)
        add_term(a00, b10, c11, 1, 1, 1)
        add_term(a00, b11, c11, 1, -1, 1)
        
        # M4 = A11(B01-B10)
        add_term(a11, b01, c00, 1, 1, 1)
        add_term(a11, b10, c00, 1, -1, 1)
        add_term(a11, b01, c10, 1, 1, 1)
        add_term(a11, b10, c10, 1, -1, 1)
        
        # M5 = (A00+A01)B11
        add_term(a00, b11, c00, 1, 1, -1)
        add_term(a01, b11, c00, 1, 1, -1)
        add_term(a00, b11, c01, 1, 1, 1)
        add_term(a01, b11, c01, 1, 1, 1)
        
        # M6 = (A10-A00)(B00+B10)
        add_term(a10, b00, c11, 1, 1, 1)
        add_term(a10, b10, c11, 1, 1, 1)
        add_term(a00, b00, c11, -1, 1, 1)
        add_term(a00, b10, c11, -1, 1, 1)
        
        # M7 = (A01-A11)(B01+B11)
        add_term(a01, b01, c00, 1, 1, 1)
        add_term(a01, b11, c00, 1, 1, 1)
        add_term(a11, b01, c00, -1, 1, 1)
        add_term(a11, b11, c00, -1, 1, 1)
    
    def standard_2x2x1(row_offset, col_offset, b_col):
        """
        Add a standard rank-4 decomposition for A[row_offset:row_offset+2, col_offset:col_offset+2]
        * B[col_offset:col_offset+2, b_col:b_col+1]
        -> C[b_col:b_col+1, row_offset:row_offset+2]
        """
        a00 = row_offset * m + col_offset
        a01 = row_offset * m + col_offset + 1
        a10 = (row_offset + 1) * m + col_offset
        a11 = (row_offset + 1) * m + col_offset + 1
        
        b0 = col_offset * p + b_col
        b1 = (col_offset + 1) * p + b_col
        
        c0 = b_col * n + row_offset
        c1 = b_col * n + row_offset + 1
        
        add_term(a00, b0, c0, 1, 1, 1)
        add_term(a01, b1, c0, 1, 1, 1)
        add_term(a10, b0, c1, 1, 1, 1)
        add_term(a11, b1, c1, 1, 1, 1)
    
    def standard_1x2x7(row, col_offset):
        """
        Add standard rank-14 decomposition for A[row, col_offset:col_offset+2]
        * B[col_offset:col_offset+2, 0:7]
        -> C[0:7, row]
        """
        a0 = row * m + col_offset
        a1 = row * m + col_offset + 1
        
        for k in range(p):
            b0 = col_offset * p + k
            b1 = (col_offset + 1) * p + k
            c = k * n + row
            
            add_term(a0, b0, c, 1, 1, 1)
            add_term(a1, b1, c, 1, 1, 1)
    
    # Process the 3x4 matrix multiplication as:
    # Block 1: rows 0-1, cols 0-1 of A (2x2) * B (2x7) -> C (7x2)
    # Block 2: rows 0-1, cols 2-3 of A (2x2) * B (2x7) -> C (7x2)
    # Block 3: row 2, cols 0-1 of A (1x2) * B (2x7) -> C (7x1)
    # Block 4: row 2, cols 2-3 of A (1x2) * B (2x7) -> C (7x1)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:7] -> C[0:7, 0:2]
    # Strassen for B cols 0-1, 2-3, 4-5
    strassen_2x2x2(0, 0, 0)
    strassen_2x2x2(0, 0, 2)
    strassen_2x2x2(0, 0, 4)
    # Standard for B col 6
    standard_2x2x1(0, 0, 6)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:7] -> C[0:7, 0:2]
    # Strassen for B cols 0-1, 2-3, 4-5
    strassen_2x2x2(0, 2, 0)
    strassen_2x2x2(0, 2, 2)
    strassen_2x2x2(0, 2, 4)
    # Standard for B col 6
    standard_2x2x1(0, 2, 6)
    
    # Block 3: A[2, 0:2] * B[0:2, 0:7] -> C[0:7, 2]
    standard_1x2x7(2, 0)
    
    # Block 4: A[2, 2:4] * B[2:4, 0:7] -> C[0:7, 2]
    standard_1x2x7(2, 2)
    
    return {"U": U, "V": V, "W": W}
