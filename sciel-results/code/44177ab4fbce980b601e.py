import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 solution:
    # Split m=4 into two blocks of size 2.
    # A = [A_L | A_R], B = [B_T; B_B]
    # C = A_L B_T + A_R B_B
    # Each term is M_{2,2,6}.
    # Split B_T (2x6) into 3 blocks of 2x2: B_T1, B_T2, B_T3.
    # A_L B_T = A_L B_T1 + A_L B_T2 + A_L B_T3.
    # Each A_L B_Ti is M_{2,2,2} (rank 7).
    # Total rank for A_L B_T is 3 * 7 = 21.
    # Same for A_R B_B.
    # Total rank 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen decomposition for M_{2,2,2}
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # U_A, U_B, U_C indices for a single M_{2,2,2} block.
    # We need to map global indices to local indices.
    
    def add_strassen_block(u_offset, v_offset, w_offset, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows: list of 2 global row indices for A
        # a_cols: list of 2 global col indices for A
        # b_rows: list of 2 global row indices for B
        # b_cols: list of 2 global col indices for B
        # c_rows: list of 2 global row indices for C (output rows)
        # c_cols: list of 2 global col indices for C (output cols)
        
        # Strassen 7 products for 2x2 matrices
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will add 7 entries to U, V, W for this block.
        
        # Local A indices: 0, 1, 2, 3 corresponding to A11, A12, A21, A22
        # Global A index: a_rows[i] * m + a_cols[j]
        
        # M1: (A11+A22)(B11+B22)
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=1, C22=1 (with signs later)
        
        # We define the 7 factors.
        # Factor 1: M1
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=1, C22=1
        
        # Factor 2: M2
        # U: A21=1, A22=1
        # V: B11=1
        # W: C21=1, C22=-1
        
        # Factor 3: M3
        # U: A11=1
        # V: B12=1, B22=-1
        # W: C12=1, C22=1
        
        # Factor 4: M4
        # U: A22=1
        # V: B21=1, B11=-1
        # W: C11=1, C21=1
        
        # Factor 5: M5
        # U: A11=1, A12=1
        # V: B22=1
        # W: C11=-1, C12=1
        
        # Factor 6: M6
        # U: A21=1, A11=-1
        # V: B11=1, B12=1
        # W: C22=1
        
        # Factor 7: M7
        # U: A12=1, A22=-1
        # V: B21=1, B22=1
        # W: C11=1, C22=1
        
        # Helper to get global index
        def g_idx(rows, cols):
            return rows * m + cols # This is wrong, m is 4.
            # A is n x m. Index is i*m + j.
            # B is m x p. Index is j*p + k.
            # C is n x p. Index is i*p + k. (Trace convention: W[k*n+i, r] -> C[i,k] is at row k*n+i? No.)
            # Task says: W[k*n+i, r]. This means W is indexed by (k, i) where k is col of C, i is row of C.
            # So W row index = k*n + i.
            
        # Let's clarify indices.
        # A: n x m. Index (i, j) -> i*m + j.
        # B: m x p. Index (j, k) -> j*p + k.
        # C: n x p. Index (i, k) -> k*n + i.
        
        # a_rows, a_cols are i, j.
        # b_rows, b_cols are j, k.
        # c_rows, c_cols are i, k.
        
        def a_idx(i, j): return i * m + j
        def b_idx(j, k): return j * p + k
        def c_idx(i, k): return k * n + i
        
        # M1
        U[a_idx(a_rows[0], a_cols[0])][u_offset] += 1 # A11
        U[a_idx(a_rows[1], a_cols[1])][u_offset] += 1 # A22
        V[b_idx(b_rows[0], b_cols[0])][u_offset] += 1 # B11
        V[b_idx(b_rows[1], b_cols[1])][u_offset] += 1 # B22
        W[c_idx(c_rows[0], c_cols[0])][u_offset] += 1 # C11
        W[c_idx(c_rows[1], c_cols[1])][u_offset] += 1 # C22
        u_offset += 1
        
        # M2
        U[a_idx(a_rows[1], a_cols[0])][u_offset] += 1 # A21
        U[a_idx(a_rows[1], a_cols[1])][u_offset] += 1 # A22
        V[b_idx(b_rows[0], b_cols[0])][u_offset] += 1 # B11
        W[c_idx(c_rows[1], c_cols[0])][u_offset] += 1 # C21
        W[c_idx(c_rows[1], c_cols[1])][u_offset] += -1 # C22
        u_offset += 1
        
        # M3
        U[a_idx(a_rows[0], a_cols[0])][u_offset] += 1 # A11
        V[b_idx(b_rows[0], b_cols[1])][u_offset] += 1 # B12
        V[b_idx(b_rows[1], b_cols[1])][u_offset] += -1 # B22
        W[c_idx(c_rows[0], c_cols[1])][u_offset] += 1 # C12
        W[c_idx(c_rows[1], c_cols[1])][u_offset] += 1 # C22
        u_offset += 1
        
        # M4
        U[a_idx(a_rows[1], a_cols[1])][u_offset] += 1 # A22
        V[b_idx(b_rows[1], b_cols[0])][u_offset] += 1 # B21
        V[b_idx(b_rows[0], b_cols[0])][u_offset] += -1 # B11
        W[c_idx(c_rows[0], c_cols[0])][u_offset] += 1 # C11
        W[c_idx(c_rows[1], c_cols[0])][u_offset] += 1 # C21
        u_offset += 1
        
        # M5
        U[a_idx(a_rows[0], a_cols[0])][u_offset] += 1 # A11
        U[a_idx(a_rows[0], a_cols[1])][u_offset] += 1 # A12
        V[b_idx(b_rows[1], b_cols[1])][u_offset] += 1 # B22
        W[c_idx(c_rows[0], c_cols[0])][u_offset] += -1 # C11
        W[c_idx(c_rows[0], c_cols[1])][u_offset] += 1 # C12
        u_offset += 1
        
        # M6
        U[a_idx(a_rows[1], a_cols[0])][u_offset] += 1 # A21
        U[a_idx(a_rows[0], a_cols[0])][u_offset] += -1 # A11
        V[b_idx(b_rows[0], b_cols[0])][u_offset] += 1 # B11
        V[b_idx(b_rows[0], b_cols[1])][u_offset] += 1 # B12
        W[c_idx(c_rows[1], c_cols[1])][u_offset] += 1 # C22
        u_offset += 1
        
        # M7
        U[a_idx(a_rows[0], a_cols[1])][u_offset] += 1 # A12
        U[a_idx(a_rows[1], a_cols[1])][u_offset] += -1 # A22
        V[b_idx(b_rows[1], b_cols[0])][u_offset] += 1 # B21
        V[b_idx(b_rows[1], b_cols[1])][u_offset] += 1 # B22
        W[c_idx(c_rows[0], c_cols[0])][u_offset] += 1 # C11
        W[c_idx(c_rows[1], c_cols[1])][u_offset] += 1 # C22
        u_offset += 1
        
        return u_offset

    # Block 1: A_L (rows 0,1; cols 0,1) * B_T (rows 0,1; cols 0,5)
    # Split B_T into 3 blocks of 2 cols: (0,1), (2,3), (4,5)
    
    a_rows = [0, 1]
    a_cols_L = [0, 1]
    b_rows_T = [0, 1]
    
    offset = 0
    # B_T1: cols 0,1
    offset = add_strassen_block(offset, 0, 0, a_rows, a_cols_L, b_rows_T, [0, 1], a_rows, [0, 1])
    # B_T2: cols 2,3
    offset = add_strassen_block(offset, 0, 0, a_rows, a_cols_L, b_rows_T, [2, 3], a_rows, [2, 3])
    # B_T3: cols 4,5
    offset = add_strassen_block(offset, 0, 0, a_rows, a_cols_L, b_rows_T, [4, 5], a_rows, [4, 5])
    
    # Block 2: A_R (rows 0,1; cols 2,3) * B_B (rows 2,3; cols 0,5)
    a_cols_R = [2, 3]
    b_rows_B = [2, 3]
    
    # B_B1: cols 0,1
    offset = add_strassen_block(offset, 0, 0, a_rows, a_cols_R, b_rows_B, [0, 1], a_rows, [0, 1])
    # B_B2: cols 2,3
    offset = add_strassen_block(offset, 0, 0, a_rows, a_cols_R, b_rows_B, [2, 3], a_rows, [2, 3])
    # B_B3: cols 4,5
    offset = add_strassen_block(offset, 0, 0, a_rows, a_cols_R, b_rows_B, [4, 5], a_rows, [4, 5])
    
    return {"U": U, "V": V, "W": W}

