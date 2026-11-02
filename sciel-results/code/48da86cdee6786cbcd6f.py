import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M2 + M4
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M7
    
    # We need to map these to the global indices.
    # Block index: 8 blocks.
    # Each block uses 7 ranks.
    
    # A is 2x4. Indices (i, j) where i in [0,1], j in [0,3].
    # B is 4x8. Indices (j, k) where j in [0,3], k in [0,7].
    # C is 2x8. Indices (i, k) where i in [0,1], k in [0,7].
    
    # Block b (0 to 7):
    # b = 0,1: A_L (j=0,1), B_T (j=0,1), k=0,1
    # b = 2,3: A_L (j=0,1), B_T (j=0,1), k=2,3
    # ...
    # Actually, B_T is 2x8. Split into 4 blocks of 2x2.
    # Block 0: A_L, B_T[:, 0:2]
    # Block 1: A_L, B_T[:, 2:4]
    # Block 2: A_L, B_T[:, 4:6]
    # Block 3: A_L, B_T[:, 6:8]
    # Block 4: A_R, B_B[:, 0:2]
    # ...
    
    # For each block, we define the local 2x2x2 Strassen decomposition.
    # Local A indices: (0,0), (0,1), (1,0), (1,1)
    # Local B indices: (0,0), (0,1), (1,0), (1,1)
    # Local C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Global mapping for a block:
    # A row i, col j_local -> Global A[i, j_offset + j_local]
    # B row j_local, col k_local -> Global B[j_offset + j_local, k_offset + k_local]
    # C row i, col k_local -> Global C[i, k_offset + k_local]
    
    # Strassen M1..M7 definitions in terms of local A, B entries.
    # We will add these to U, V, W.
    
    # Helper to add a Strassen block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset, rank_offset):
        # a_row_offset is always 0 or 1 (since n=2, we don't split rows of A)
        # Wait, A is 2x4. We split cols.
        # So a_row_offset is 0 for A_L, 1 for A_R? No, A_L and A_R are both 2x2.
        # A_L uses rows 0,1 of A. A_R uses rows 0,1 of A.
        # So a_row_offset is 0 for both.
        # a_col_offset is 0 for A_L, 2 for A_R.
        
        # b_row_offset is 0 for B_T, 2 for B_B.
        # b_col_offset is 0, 2, 4, 6.
        
        # c_col_offset is 0, 2, 4, 6.
        
        # Local indices
        # A: (0,0) -> (0, a_col_offset)
        # A: (0,1) -> (0, a_col_offset+1)
        # A: (1,0) -> (1, a_col_offset)
        # A: (1,1) -> (1, a_col_offset+1)
        
        # B: (0,0) -> (b_row_offset, b_col_offset)
        # B: (0,1) -> (b_row_offset, b_col_offset+1)
        # B: (1,0) -> (b_row_offset+1, b_col_offset)
        # B: (1,1) -> (b_row_offset+1, b_col_offset+1)
        
        # C: (0,0) -> (0, c_col_offset)
        # C: (0,1) -> (0, c_col_offset+1)
        # C: (1,0) -> (1, c_col_offset)
        # C: (1,1) -> (1, c_col_offset+1)
        
        # Strassen M1
        # U: A00, A11
        # V: B00, B11
        # W: C00 (coeff 1), C11 (coeff 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[0*a_row_offset + 0, rank_offset + 0] += 1 # A00
        U[1*a_row_offset + 1, rank_offset + 0] += 1 # A11
        V[0*b_row_offset + 0, rank_offset + 0] += 1 # B00
        V[1*b_row_offset + 1, rank_offset + 0] += 1 # B11
        W[0*c_col_offset + 0, rank_offset + 0] += 1 # C00
        W[1*c_col_offset + 1, rank_offset + 0] += 1 # C11
        
        # M2 = (A10+A11)B00
        U[1*a_row_offset + 0, rank_offset + 1] += 1 # A10
        U[1*a_row_offset + 1, rank_offset + 1] += 1 # A11
        V[0*b_row_offset + 0, rank_offset + 1] += 1 # B00
        W[1*c_col_offset + 0, rank_offset + 1] += 1 # C10
        W[0*c_col_offset + 0, rank_offset + 1] -= 1 # C00 (from -M2)
        
        # M3 = A00(B10-B00)
        U[0*a_row_offset + 0, rank_offset + 2] += 1 # A00
        V[1*b_row_offset + 0, rank_offset + 2] += 1 # B10
        V[0*b_row_offset + 0, rank_offset + 2] -= 1 # -B00
        W[0*c_col_offset + 1, rank_offset + 2] += 1 # C01
        W[1*c_col_offset + 1, rank_offset + 2] -= 1 # C11 (from -M3)
        
        # M4 = A11(B01-B11)
        U[1*a_row_offset + 1, rank_offset + 3] += 1 # A11
        V[0*b_row_offset + 1, rank_offset + 3] += 1 # B01
        V[1*b_row_offset + 1, rank_offset + 3] -= 1 # -B11
        W[0*c_col_offset + 0, rank_offset + 3] += 1 # C00
        W[1*c_col_offset + 1, rank_offset + 3] += 1 # C11
        
        # M5 = (A00+A01)B11
        U[0*a_row_offset + 0, rank_offset + 4] += 1 # A00
        U[0*a_row_offset + 1, rank_offset + 4] += 1 # A01
        V[1*b_row_offset + 1, rank_offset + 4] += 1 # B11
        W[0*c_col_offset + 1, rank_offset + 4] += 1 # C01
        
        # M6 = (A10-A00)(B00+B10)
        U[1*a_row_offset + 0, rank_offset + 5] += 1 # A10
        U[0*a_row_offset + 0, rank_offset + 5] -= 1 # -A00
        V[0*b_row_offset + 0, rank_offset + 5] += 1 # B00
        V[1*b_row_offset + 0, rank_offset + 5] += 1 # B10
        W[1*c_col_offset + 0, rank_offset + 5] += 1 # C10
        
        # M7 = (A01-A11)(B01+B11)
        U[0*a_row_offset + 1, rank_offset + 6] += 1 # A01
        U[1*a_row_offset + 1, rank_offset + 6] -= 1 # -A11
        V[0*b_row_offset + 1, rank_offset + 6] += 1 # B01
        V[1*b_row_offset + 1, rank_offset + 6] += 1 # B11
        W[1*c_col_offset + 1, rank_offset + 6] += 1 # C11
        
    # Loop over 8 blocks
    # 4 blocks for A_L (a_col_offset=0), 4 for A_R (a_col_offset=2)
    # For each, 4 blocks for B cols (b_col_offset=0,2,4,6)
    # B rows: A_L uses B_T (b_row_offset=0), A_R uses B_B (b_row_offset=2)
    
    rank_idx = 0
    for a_col_off in [0, 2]:
        b_row_off = 0 if a_col_off == 0 else 2
        for b_col_off in [0, 2, 4, 6]:
            c_col_off = b_col_off
            add_strassen_block(0, a_col_off, b_row_off, b_col_off, c_col_off, rank_idx)
            rank_idx += 7
            
    return {"U": U, "V": V, "W": W}
