import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 34
    # by splitting the 6 columns into two groups of 3.
    # Each group requires rank 17 for 2x3x3.
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        
    # We need to implement the rank 17 decomposition for 2x3x3.
    # 2x3x3: A (2x3), B (3x3), C (2x3)
    # Split B into B_left (3x2) and B_right (3x1).
    # C = A * B_left + A * B_right.
    # A * B_left is 2x3 * 3x2.
    # Split A into A_top (2x2) and A_bot (2x1)? No, A is 2x3.
    # Let A = [A_1, A_2] where A_1 is 2x2 (cols 0,1) and A_2 is 2x1 (col 2).
    # Let B_left = [B_1; B_2] where B_1 is 2x2 (rows 0,1) and B_2 is 1x2 (row 2).
    # C_left = A_1 * B_1 + A_2 * B_2.
    # A_1 * B_1 is 2x2 * 2x2. Rank 7 (Strassen).
    # A_2 * B_2 is 2x1 * 1x2. Rank 4.
    # Total for C_left is 11.
    # C_right = A * B_right.
    # A is 2x3, B_right is 3x1. Rank 6.
    # Total for 2x3x3 is 11 + 6 = 17.
    
    # We will replicate this for the first 3 columns and the last 3 columns.
    
    # Strassen 2x2x2 decomposition
    # U indices for 2x2: (0,0), (0,1), (1,0), (1,1)
    # V indices for 2x2: (0,0), (0,1), (1,0), (1,1)
    # W indices for 2x2: (0,0), (0,1), (1,0), (1,1)
    
    def strassen_2x2(u_base, v_base, w_base, start_col):
        # u_base: index in U for (0,0)
        # v_base: index in V for (0,0)
        # w_base: index in W for (0,0)
        # Indices:
        # U: i*m + j. m=2 for this subproblem? No, m is global.
        # We are mapping local indices to global.
        # Local A is 2x2. Global A is 2x3.
        # Local (i, j) -> Global (i, j)
        # Local B is 2x2. Global B is 3x2 (for left part).
        # Local (j, k) -> Global (j, k)
        # Local C is 2x2. Global C is 2x2.
        # Local (k, i) -> Global (k, i)
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add these to U, V, W.
        # Each term is a rank-1 tensor.
        
        # Term 1: (A00+A11) * (B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11 (with signs)
        
        # This is getting complex to hardcode with signs.
        # I will just use the schoolbook for the sub-blocks to ensure correctness, 
        # and then optimize if time permits.
        # Schoolbook for 2x2x2 is rank 4.
        # 4 + 4 (for A2*B2) + 6 (for A*Br) = 14.
        # 14 * 2 = 28.
        # This is better than 34!
        
        # Let's use schoolbook for the sub-blocks.
        # 2x3x3:
        # A1 (2x2) * B1 (2x2) -> Rank 4
        # A2 (2x1) * B2 (1x2) -> Rank 4
        # A (2x3) * B3 (3x1) -> Rank 6
        # Total 14.
        # Total for 2x3x6: 28.
        
        pass

    # Let's implement the rank 28 construction.
    # Block 1: C[:, 0:3] = A * B[:, 0:3]
    # Block 2: C[:, 3:6] = A * B[:, 3:6]
    
    def add_2x3x3_block(col_offset, b_col_offset):
        # col_offset: start column in U,V,W
        # b_col_offset: start column in B (0 or 3)
        
        # Sub-block 1: A[:, 0:2] * B[0:2, b_col_offset:b_col_offset+2]
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0, bco), (0, bco+1), (1, bco), (1, bco+1)
        # C indices: (bco, 0), (bco+1, 0), (bco, 1), (bco+1, 1)
        
        # Schoolbook for 2x2x2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = i * 3 + j # Global A index
                    v_idx = j * 6 + (b_col_offset + k) # Global B index
                    w_idx = (b_col_offset + k) * 2 + i # Global C index
                    add_term(u_idx, v_idx, w_idx, col_offset)
                    col_offset += 1
        # This used 4 terms.
        
        # Sub-block 2: A[:, 2] * B[2, b_col_offset:b_col_offset+2]
        # A index: (0,2), (1,2)
        # B indices: (2, bco), (2, bco+1)
        # C indices: (bco, 0), (bco+1, 0), (bco, 1), (bco+1, 1)
        
        for i in range(2):
            for k in range(2):
                u_idx = i * 3 + 2
                v_idx = 2 * 6 + (b_col_offset + k)
                w_idx = (b_col_offset + k) * 2 + i
                add_term(u_idx, v_idx, w_idx, col_offset)
                col_offset += 1
        # This used 4 terms.
        
        # Sub-block 3: A[:, 0:3] * B[0:3, b_col_offset+2]
        # A indices: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
        # B index: (0, bco+2), (1, bco+2), (2, bco+2)
        # C indices: (bco+2, 0), (bco+2, 1)
        
        for i in range(2):
            for j in range(3):
                u_idx = i * 3 + j
                v_idx = j * 6 + (b_col_offset + 2)
                w_idx = (b_col_offset + 2) * 2 + i
                add_term(u_idx, v_idx, w_idx, col_offset)
                col_offset += 1
        # This used 6 terms.
        
        return col_offset

    # Block 1
    next_col = add_2x3x3_block(0, 0)
    # Block 2
    next_col = add_2x3x3_block(next_col, 3)
    
    return {"U": U, "V": V, "W": W}

