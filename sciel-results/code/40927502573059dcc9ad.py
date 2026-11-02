import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen multipliers for 2x2 matrices
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x01+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x11-x10)(y10+y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We define the mapping for each of the 4 blocks.
    # Block 0: A_1 B_11 -> C_11
    # Block 1: A_2 B_21 -> C_11
    # Block 2: A_1 B_12 -> C_21
    # Block 3: A_2 B_22 -> C_21
    
    # Indices in A (2x4): row i, col j. Flat index i*m+j = i*4+j.
    # Indices in B (4x4): row j, col k. Flat index j*p+k = j*4+k.
    # Indices in C (2x4) via W (4x2): row k, col i. Flat index k*n+i = k*2+i.
    
    # Strassen terms
    # For a block with A rows a_r, A cols a_c, B rows b_r, B cols b_c, C rows c_r, C cols c_c
    # We need to map local 2x2 indices to global flat indices.
    
    def add_strassen_block(base_col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows: list of 2 global row indices in A (0,1)
        # a_cols: list of 2 global col indices in A (0,1,2,3)
        # b_rows: list of 2 global row indices in B (0,1,2,3)
        # b_cols: list of 2 global col indices in B (0,1,2,3)
        # c_rows: list of 2 global row indices in C (0,1,2,3) -> these are k
        # c_cols: list of 2 global col indices in C (0,1) -> these are i
        
        # Local indices 0,1
        # A local (r, c) -> global (a_rows[r], a_cols[c]) -> flat a_rows[r]*4 + a_cols[c]
        # B local (r, c) -> global (b_rows[r], b_cols[c]) -> flat b_rows[r]*4 + b_cols[c]
        # C local (r, c) -> global (c_rows[r], c_cols[c]) -> flat c_rows[r]*2 + c_cols[c]
        
        # Strassen M1..M7
        # M1: (A00+A11)(B00+B11)
        # M2: (A01+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A11-A10)(B10+B11)
        
        # We will add contributions to U, V, W for each of the 7 multipliers.
        
        # M1
        col = base_col
        # U: A00, A11
        U[a_rows[0]*4 + a_cols[0], col] += 1
        U[a_rows[1]*4 + a_cols[1], col] += 1
        # V: B00, B11
        V[b_rows[0]*4 + b_cols[0], col] += 1
        V[b_rows[1]*4 + b_cols[1], col] += 1
        # W: Z00(+1), Z11(+1)
        W[c_rows[0]*2 + c_cols[0], col] += 1
        W[c_rows[1]*2 + c_cols[1], col] += 1
        
        # M2
        col = base_col + 1
        # U: A01, A11
        U[a_rows[0]*4 + a_cols[1], col] += 1
        U[a_rows[1]*4 + a_cols[1], col] += 1
        # V: B00
        V[b_rows[0]*4 + b_cols[0], col] += 1
        # W: Z10(+1)
        W[c_rows[1]*2 + c_cols[0], col] += 1
        
        # M3
        col = base_col + 2
        # U: A00
        U[a_rows[0]*4 + a_cols[0], col] += 1
        # V: B01, -B11
        V[b_rows[0]*4 + b_cols[1], col] += 1
        V[b_rows[1]*4 + b_cols[1], col] += -1
        # W: Z01(+1), Z11(+1)
        W[c_rows[0]*2 + c_cols[1], col] += 1
        W[c_rows[1]*2 + c_cols[1], col] += 1
        
        # M4
        col = base_col + 3
        # U: A11
        U[a_rows[1]*4 + a_cols[1], col] += 1
        # V: B10, -B00
        V[b_rows[1]*4 + b_cols[0], col] += 1
        V[b_rows[0]*4 + b_cols[0], col] += -1
        # W: Z00(+1), Z10(+1)
        W[c_rows[0]*2 + c_cols[0], col] += 1
        W[c_rows[1]*2 + c_cols[0], col] += 1
        
        # M5
        col = base_col + 4
        # U: A00, A01
        U[a_rows[0]*4 + a_cols[0], col] += 1
        U[a_rows[0]*4 + a_cols[1], col] += 1
        # V: B11
        V[b_rows[1]*4 + b_cols[1], col] += 1
        # W: Z00(-1), Z01(+1)
        W[c_rows[0]*2 + c_cols[0], col] += -1
        W[c_rows[0]*2 + c_cols[1], col] += 1
        
        # M6
        col = base_col + 5
        # U: A10, -A00
        U[a_rows[1]*4 + a_cols[0], col] += 1
        U[a_rows[0]*4 + a_cols[0], col] += -1
        # V: B00, B01
        V[b_rows[0]*4 + b_cols[0], col] += 1
        V[b_rows[0]*4 + b_cols[1], col] += 1
        # W: Z11(+1)
        W[c_rows[1]*2 + c_cols[1], col] += 1
        
        # M7
        col = base_col + 6
        # U: A11, -A10
        U[a_rows[1]*4 + a_cols[1], col] += 1
        U[a_rows[1]*4 + a_cols[0], col] += -1
        # V: B10, B11
        V[b_rows[1]*4 + b_cols[0], col] += 1
        V[b_rows[1]*4 + b_cols[1], col] += 1
        # W: Z00(+1)
        W[c_rows[0]*2 + c_cols[0], col] += 1

    # Block 0: A_1 (rows 0,1; cols 0,1) * B_11 (rows 0,1; cols 0,1) -> C_11 (rows 0,1; cols 0,1)
    add_strassen_block(0, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 1: A_2 (rows 0,1; cols 2,3) * B_21 (rows 2,3; cols 0,1) -> C_11 (rows 0,1; cols 0,1)
    add_strassen_block(7, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # Block 2: A_1 (rows 0,1; cols 0,1) * B_12 (rows 0,1; cols 2,3) -> C_21 (rows 2,3; cols 0,1)
    add_strassen_block(14, [0,1], [0,1], [0,1], [2,3], [2,3], [0,1])
    
    # Block 3: A_2 (rows 0,1; cols 2,3) * B_22 (rows 2,3; cols 2,3) -> C_21 (rows 2,3; cols 0,1)
    add_strassen_block(21, [0,1], [2,3], [2,3], [2,3], [2,3], [0,1])
    
    return {"U": U, "V": V, "W": W}
