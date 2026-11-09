import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U: (4, 7) for A (2x2)
    # V: (4, 7) for B (2x2)
    # W: (4, 7) for C (2x2)
    
    # Strassen terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3 (Note: B is usually indexed B[j,k] where j is row, k is col. 
    # In our tensor convention V[j*p+k, r]. For 2x2, p=2.
    # V[0, r] = B00, V[1, r] = B01, V[2, r] = B10, V[3, r] = B11.
    # W: C[i,k]. W[0, r] = C00, W[1, r] = C01, W[2, r] = C10, W[3, r] = C11.
    
    U_strassen = np.zeros((4, 7), dtype=int)
    V_strassen = np.zeros((4, 7), dtype=int)
    W_strassen = np.zeros((4, 7), dtype=int)
    
    # M1
    U_strassen[0, 0] = 1 # A00
    U_strassen[3, 0] = 1 # A11
    V_strassen[0, 0] = 1 # B00
    V_strassen[3, 0] = 1 # B11
    W_strassen[0, 0] = 1 # C00
    W_strassen[3, 0] = 1 # C11
    
    # M2
    U_strassen[2, 1] = 1 # A10
    U_strassen[3, 1] = 1 # A11
    V_strassen[0, 1] = 1 # B00
    W_strassen[2, 1] = 1 # C10
    W_strassen[3, 1] = 1 # C11
    
    # M3
    U_strassen[0, 2] = 1 # A00
    V_strassen[2, 2] = 1 # B10
    V_strassen[3, 2] = -1 # B11
    W_strassen[1, 2] = 1 # C01
    W_strassen[2, 2] = 1 # C10
    
    # M4
    U_strassen[3, 3] = 1 # A11
    V_strassen[1, 3] = 1 # B01
    V_strassen[0, 3] = -1 # B00
    W_strassen[0, 3] = 1 # C00
    W_strassen[2, 3] = 1 # C10
    
    # M5
    U_strassen[0, 4] = 1 # A00
    U_strassen[1, 4] = 1 # A01
    V_strassen[3, 4] = 1 # B11
    W_strassen[0, 4] = -1 # C00
    W_strassen[1, 4] = 1 # C01
    
    # M6
    U_strassen[2, 5] = 1 # A10
    U_strassen[0, 5] = -1 # A00
    V_strassen[0, 5] = 1 # B00
    V_strassen[1, 5] = 1 # B01
    W_strassen[3, 5] = 1 # C11
    
    # M7
    U_strassen[1, 6] = 1 # A01
    U_strassen[3, 6] = -1 # A11
    V_strassen[2, 6] = 1 # B10
    V_strassen[3, 6] = 1 # B11
    W_strassen[0, 6] = 1 # C00
    W_strassen[1, 6] = -1 # C01
    
    # Now build the 4x4x6 decomposition
    # Rank = 4 (sub-problems) * 3 (B blocks) * 7 (Strassen) = 84
    
    rank = 84
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Sub-problem definitions:
    # (a_row_start, b_row_start, c_row_start)
    # A is 4x4, B is 4x6, C is 4x6
    # A rows: 0-1 (top), 2-3 (bot)
    # B rows: 0-1 (top), 2-3 (bot)
    # C rows: 0-1 (top), 2-3 (bot)
    
    # 1. A_top (0-1) * B_top (0-1) -> C_top (0-1)
    # 2. A_top (0-1) * B_bot (2-3) -> C_top (0-1)
    # 3. A_bot (2-3) * B_top (0-1) -> C_bot (2-3)
    # 4. A_bot (2-3) * B_bot (2-3) -> C_bot (2-3)
    
    sub_problems = [
        (0, 0, 0), # A 0-1, B 0-1, C 0-1
        (0, 2, 0), # A 0-1, B 2-3, C 0-1
        (2, 0, 2), # A 2-3, B 0-1, C 2-3
        (2, 2, 2), # A 2-3, B 2-3, C 2-3
    ]
    
    col_offset = 0
    for sp in sub_problems:
        a_start, b_start, c_start = sp
        
        # B is 2x6 in this sub-problem context? No, B is 2x6 globally, but we split into 2x2 blocks.
        # The "B" for the 2x2x2 Strassen is 2x2.
        # We have 3 such blocks for the 2x6 width.
        for b_block in range(3):
            # b_block 0: cols 0-1
            # b_block 1: cols 2-3
            # b_block 2: cols 4-5
            
            # For each Strassen term
            for s in range(7):
                col = col_offset + s
                
                # Fill U
                # U index: a_start + row_in_block
                for r in range(4):
                    val = U_strassen[r, s]
                    if val != 0:
                        U[a_start + r, col] = val
                
                # Fill V
                # V index: (b_start + row_in_block)*p + (b_block*2 + col_in_block)
                # p=6
                for r in range(4):
                    val = V_strassen[r, s]
                    if val != 0:
                        b_row = b_start + r
                        # r is 0-3 corresponding to B00, B01, B10, B11 in local 2x2
                        # Local B row: r // 2? No.
                        # V_strassen indices:
                        # 0: B00 (row 0, col 0)
                        # 1: B01 (row 0, col 1)
                        # 2: B10 (row 1, col 0)
                        # 3: B11 (row 1, col 1)
                        
                        local_row = r // 2
                        local_col = r % 2
                        
                        global_b_row = b_start + local_row
                        global_b_col = b_block * 2 + local_col
                        
                        v_idx = global_b_row * p + global_b_col
                        V[v_idx, col] = val
                
                # Fill W
                # W index: (c_block*2 + col_in_block)*n + (c_start + row_in_block)
                # W[k*n+i, r] where k is col of C, i is row of C.
                # W_strassen indices:
                # 0: C00 (row 0, col 0)
                # 1: C01 (row 0, col 1)
                # 2: C10 (row 1, col 0)
                # 3: C11 (row 1, col 1)
                
                for r in range(4):
                    val = W_strassen[r, s]
                    if val != 0:
                        local_row = r // 2
                        local_col = r % 2
                        
                        global_c_row = c_start + local_row
                        global_c_col = b_block * 2 + local_col # C cols correspond to B cols in this decomposition?
                        # Yes, C is 4x6. The block structure of C matches B's column blocks.
                        
                        w_idx = global_c_col * n + global_c_row
                        W[w_idx, col] = val
                        
                col_offset += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
