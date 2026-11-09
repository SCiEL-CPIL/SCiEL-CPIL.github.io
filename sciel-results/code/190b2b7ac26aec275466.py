import time

DIMENSIONS = [2, 4, 7]

def _schoolbook_decomp():
    """Standard (schoolbook) decomposition of 2x4x7 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication using Strassen.
    
    A = [[a, b], [c, d]] (2x2), B = [[e, f], [g, h]] (2x2), C = A*B (2x2)
    
    Strassen's 7 multiplications:
      M1 = (c+d)*e
      M2 = (c+d)*(f-h)
      M3 = a*(e-f)
      M4 = d*(g-h)
      M5 = (a+b)*h
      M6 = (c-a)*(e+f)
      M7 = (b-d)*(g+h)
    
    Then:
      C[0,0] = M5 + M4 + M1 - M7
      C[0,1] = M3 + M5
      C[1,0] = M2 + M4
      C[1,1] = M1 - M3 + M6 + M7
    
    Each term is a triple (u_vec, v_vec, w_vec) where:
      u_vec is in R^4 (indexing A's entries: [a, b, c, d])
      v_vec is in R^4 (indexing B's entries: [e, f, g, h])
      w_vec is in R^4 (indexing C's entries: [C00, C01, C10, C11])
    
    The decomposition: sum_t u_t * v_t * w_t = tensor
    """
    # A indices: a=0, b=1, c=2, d=3
    # B indices: e=0, f=1, g=2, h=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    terms = []
    
    # M1 = (c+d)*e : u=[0,0,1,1], v=[1,0,0,0], contributes to C00(+1), C11(+1)
    terms.append(([0,0,1,1], [1,0,0,0], [1,0,0,1]))
    
    # M2 = (c+d)*(f-h) : u=[0,0,1,1], v=[0,1,0,-1], contributes to C10(+1), C11(-1)
    terms.append(([0,0,1,1], [0,1,0,-1], [0,0,1,-1]))
    
    # M3 = a*(e-f) : u=[1,0,0,0], v=[1,-1,0,0], contributes to C00(-1), C01(+1)
    terms.append(([1,0,0,0], [1,-1,0,0], [-1,1,0,0]))
    
    # M4 = d*(g-h) : u=[0,0,0,1], v=[0,0,1,-1], contributes to C00(+1), C10(+1)
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    
    # M5 = (a+b)*h : u=[1,1,0,0], v=[0,0,0,1], contributes to C00(+1), C01(+1)
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    
    # M6 = (c-a)*(e+f) : u=[-1,0,1,0], v=[1,1,0,0], contributes to C11(+1)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (b-d)*(g+h) : u=[0,1,0,-1], v=[0,0,1,1], contributes to C00(-1), C11(+1)
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,1]))
    
    return terms

def _block_decomp_2x4x7():
    """
    Block decomposition of 2x4x7 matrix multiplication.
    
    A is 2x4, B is 4x7, C is 2x7.
    
    Partition the middle dimension (4) into two groups of 2:
      A = [A1 | A2] where A1, A2 are 2x2
      B = [B1; B2] where B1, B2 are 2x7
    
    Then C = A1*B1 + A2*B2
    
    Further partition the output dimension (7) into:
      Columns 0-1: 2 cols
      Columns 2-3: 2 cols  
      Columns 4-5: 2 cols
      Column 6: 1 col
    
    So:
      C[:, 0:2] = A1*B1[:, 0:2] + A2*B2[:, 0:2]  (two 2x2x2 mults)
      C[:, 2:4] = A1*B1[:, 2:4] + A2*B2[:, 2:4]  (two 2x2x2 mults)
      C[:, 4:6] = A1*B1[:, 4:6] + A2*B2[:, 4:6]  (two 2x2x2 mults)
      C[:, 6:7] = A1*B1[:, 6:7] + A2*B2[:, 6:7]  (two 2x2x1 mults)
    
    Each 2x2x2 uses Strassen rank 7.
    Each 2x2x1 uses schoolbook rank 4.
    
    Total rank: 6*7 + 2*4 = 42 + 8 = 50
    """
    n, m, p = 2, 4, 7
    
    # U dimension: n*m = 8 (A entries: row i, col j -> index i*m+j)
    #   A1 entries: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3  [cols 0,1]
    #   A2 entries: (0,2)->4, (0,3)->5, (1,2)->6, (1,3)->7  [cols 2,3]
    
    # V dimension: m*p = 28 (B entries: row j, col k -> index j*p+k)
    #   B1 entries (rows 0,1; cols 0-6): indices 0-6, 7-13
    #     B1[:, 0:2]: (0,0)->0, (0,1)->1, (1,0)->7, (1,1)->8
    #     B1[:, 2:4]: (0,2)->2, (0,3)->3, (1,2)->9, (1,3)->10
    #     B1[:, 4:6]: (0,4)->4, (0,5)->5, (1,4)->11, (1,5)->12
    #     B1[:, 6:7]: (0,6)->6, (1,6)->13
    #   B2 entries (rows 2,3; cols 0-6): indices 14-20, 21-27
    #     B2[:, 0:2]: (2,0)->14, (2,1)->15, (3,0)->21, (3,1)->22
    #     B2[:, 2:4]: (2,2)->16, (2,3)->17, (3,2)->23, (3,3)->24
    #     B2[:, 4:6]: (2,4)->18, (2,5)->19, (3,4)->25, (3,5)->26
    #     B2[:, 6:7]: (2,6)->20, (3,6)->27
    
    # W dimension: p*n = 14 (C entries: row k, col i -> index k*n+i)
    #   C[:, 0:2]: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   C[:, 2:4]: (2,0)->4, (2,1)->5, (3,0)->6, (3,1)->7
    #   C[:, 4:6]: (4,0)->8, (4,1)->9, (5,0)->10, (5,1)->11
    #   C[:, 6:7]: (6,0)->12, (6,1)->13
    
    strassen_terms = _strassen_2x2x2_terms()
    
    # Build the full decomposition
    U_cols = []  # list of columns, each is a vector of length 8
    V_cols = []  # list of columns, each is a vector of length 28
    W_cols = []  # list of columns, each is a vector of length 14
    
    # For each 2x2x2 block:
    #   Block (A_part, B_part, C_part)
    #   A_part: 2x2 submatrix of A
    #   B_part: 2x2 submatrix of B  
    #   C_part: 2x2 submatrix of C
    
    # Define the 6 2x2x2 blocks:
    # (a_part, b_row_start, b_col_start, c_col_start)
    # A parts: A1 (cols 0-1) or A2 (cols 2-3)
    # B parts: B1 (rows 0-1) or B2 (rows 2-3), with column offset
    # C parts: output columns
    
    blocks_2x2x2 = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        # A1 * B1[:, 0:2] -> C[:, 0:2]
        (0, 0, 0, 0),
        # A2 * B2[:, 0:2] -> C[:, 0:2]
        (2, 14, 0, 0),
        # A1 * B1[:, 2:4] -> C[:, 2:4]
        (0, 0, 2, 2),
        # A2 * B2[:, 2:4] -> C[:, 2:4]
        (2, 14, 2, 2),
        # A1 * B1[:, 4:6] -> C[:, 4:6]
        (0, 0, 4, 4),
        # A2 * B2[:, 4:6] -> C[:, 4:6]
        (2, 14, 4, 4),
    ]
    
    for a_col_off, b_row_off, b_col_off, c_col_off in blocks_2x2x2:
        for u_t, v_t, w_t in strassen_terms:
            U_vec = [0] * 8
            V_vec = [0] * 28
            W_vec = [0] * 14
            
            # u_t is [a, b, c, d] for the 2x2 A submatrix
            # Map to full A indices:
            #   a -> A[a_col_off + 0] (row 0, col a_col_off)
            #   b -> A[a_col_off + 1] (row 0, col a_col_off+1)
            #   c -> A[2 + a_col_off + 0] (row 1, col a_col_off)
            #   d -> A[2 + a_col_off + 1] (row 1, col a_col_off+1)
            U_vec[a_col_off + 0] = u_t[0]   # a
            U_vec[a_col_off + 1] = u_t[1]   # b
            U_vec[2 + a_col_off + 0] = u_t[2]  # c
            U_vec[2 + a_col_off + 1] = u_t[3]  # d
            
            # v_t is [e, f, g, h] for the 2x2 B submatrix
            # B is 4x7, indexed as j*p+k = j*7+k
            # B_row_off is the starting row (0 for B1, 2 for B2, but in flat index: 0 or 14)
            # Actually b_row_off is already in flat index space (0 or 14)
            # B submatrix rows: b_row_off//7 and b_row_off//7 + 1
            # B submatrix cols: b_col_off, b_col_off+1
            # 
            # e = B[row1, col1] = B[b_row_off//7, b_col_off] -> flat: b_row_off + b_col_off
            # f = B[row1, col2] = B[b_row_off//7, b_col_off+1] -> flat: b_row_off + b_col_off + 1
            # g = B[row2, col1] = B[b_row_off//7+1, b_col_off] -> flat: b_row_off + 7 + b_col_off
            # h = B[row2, col2] = B[b_row_off//7+1, b_col_off+1] -> flat: b_row_off + 7 + b_col_off + 1
            
            V_vec[b_row_off + b_col_off] = v_t[0]       # e
            V_vec[b_row_off + b_col_off + 1] = v_t[1]   # f
            V_vec[b_row_off + 7 + b_col_off] = v_t[2]   # g
            V_vec[b_row_off + 7 + b_col_off + 1] = v_t[3]  # h
            
            # w_t is [C00, C01, C10, C11] for the 2x2 C submatrix
            # C is 2x7, indexed as k*n+i = k*2+i
            # C submatrix cols: c_col_off, c_col_off+1
            # 
            # C00 = C[0, c_col_off] -> flat: 0*2 + c_col_off = c_col_off
            # C01 = C[0, c_col_off+1] -> flat: 0*2 + c_col_off+1 = c_col_off+1
            # C10 = C[1, c_col_off] -> flat: 1*2 + c_col_off = 2 + c_col_off
            # C11 = C[1, c_col_off+1] -> flat: 1*2 + c_col_off+1 = 3 + c_col_off
            
            W_vec[c_col_off] = w_t[0]       # C00
            W_vec[c_col_off + 1] = w_t[1]   # C01
            W_vec[2 + c_col_off] = w_t[2]   # C10
            W_vec[2 + c_col_off + 1] = w_t[3]  # C11
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
    
    # Now the two 2x2x1 blocks:
    # A1 * B1[:, 6:7] -> C[:, 6:7]
    # A2 * B2[:, 6:7] -> C[:, 6:7]
    # 
    # For 2x2x1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #             C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # 
    # Schoolbook rank 4:
    #   Term 1: A[0,0]*B[0,0] -> C[0,0]
    #   Term 2: A[0,1]*B[1,0] -> C[0,0]
    #   Term 3: A[1,0]*B[0,0] -> C[1,0]
    #   Term 4: A[1,1]*B[1,0] -> C[1,0]
    
    blocks_2x2x1 = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 6, 6),   # A1 * B1[:, 6] -> C[:, 6]
        (2, 14, 6, 6),  # A2 * B2[:, 6] -> C[:, 6]
    ]
    
    for a_col_off, b_row_off, b_col_off, c_col_off in blocks_2x2x1:
        # Term 1: A[0, a_col_off] * B[b_row_off//7, b_col_off] -> C[0, c_col_off]
        U_vec = [0] * 8
        V_vec = [0] * 28
        W_vec = [0] * 14
        U_vec[a_col_off] = 1
        V_vec[b_row_off + b_col_off] = 1
        W_vec[c_col_off] = 1
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        
        # Term 2: A[0, a_col_off+1] * B[b_row_off//7+1, b_col_off] -> C[0, c_col_off]
        U_vec = [0] * 8
        V_vec = [0] * 28
        W_vec = [0] * 14
        U_vec[a_col_off + 1] = 1
        V_vec[b_row_off + 7 + b_col_off] = 1
        W_vec[c_col_off] = 1
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        
        # Term 3: A[1, a_col_off] * B[b_row_off//7, b_col_off] -> C[1, c_col_off]
        U_vec = [0] * 8
        V_vec = [0] * 28
        W_vec = [0] * 14
        U_vec[2 + a_col_off] = 1
        V_vec[b_row_off + b_col_off] = 1
        W_vec[2 + c_col_off] = 1
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        
        # Term 4: A[1, a_col_off+1] * B[b_row_off//7+1, b_col_off] -> C[1, c_col_off]
        U_vec = [0] * 8
        V_vec = [0] * 28
        W_vec = [0] * 14
        U_vec[2 + a_col_off + 1] = 1
        V_vec[b_row_off + 7 + b_col_off] = 1
        W_vec[2 + c_col_off] = 1
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
    
    rank = len(U_cols)
    
    # Convert to row-major format: U[row][col], V[row][col], W[row][col]
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(14)]
    
    for col in range(rank):
        for row in range(8):
            U[row][col] = U_cols[col][row]
        for row in range(28):
            V[row][col] = V_cols[col][row]
        for row in range(14):
            W[row][col] = W_cols[col][row]
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 120.0
    
    try:
        # Try the block decomposition first
        result = _block_decomp_2x4x7()
        
        # Verify correctness using numpy
        import numpy as np
        
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        n, m, p = 2, 4, 7
        rank = U.shape[1]
        
        # Build the tensor from the decomposition:
        # T[i*m+j, j*p+k, k*n+i] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # Check a few entries
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    err = abs(expected - actual)
                    max_err = max(max_err, err)
        
        # Also check that off-diagonal entries are 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                expected = 0.0
                                actual = np.dot(U[i2*m+j2, :], np.multiply(V[j2*p+k2, :], W[k2*n+i2, :]))
                                err = abs(expected - actual)
                                max_err = max(max_err, err)
        
        if max_err < 1e-10:
            return result
        else:
            # Fallback to schoolbook
            return _schoolbook_decomp()
            
    except Exception as e:
        return _schoolbook_decomp()
