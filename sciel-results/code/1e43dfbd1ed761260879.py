import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # 1. Define M2 Strassen Tensor (Rank 7)
    # A indices: 0..3 (0=A00, 1=A01, 2=A10, 3=A11)
    # B indices: 0..3 (0=B00, 1=B01, 2=B10, 3=B11)
    # C indices: 0..3 (0=C00, 1=C01, 2=C10, 3=C11)
    
    U_m2 = np.zeros((4, 7), dtype=int)
    V_m2 = np.zeros((4, 7), dtype=int)
    W_m2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U_m2[0, 0] = 1; U_m2[3, 0] = 1
    V_m2[0, 0] = 1; V_m2[3, 0] = 1
    W_m2[0, 0] = 1; W_m2[3, 0] = 1
    
    # M2: (A10+A11)B00
    U_m2[2, 1] = 1; U_m2[3, 1] = 1
    V_m2[0, 1] = 1
    W_m2[1, 1] = 1; W_m2[2, 1] = 1
    
    # M3: A00(B01-B11)
    U_m2[0, 2] = 1
    V_m2[1, 2] = 1; V_m2[3, 2] = -1
    W_m2[1, 2] = 1; W_m2[3, 2] = -1
    
    # M4: A11(B10-B00)
    U_m2[3, 3] = 1
    V_m2[2, 3] = 1; V_m2[0, 3] = -1
    W_m2[0, 3] = 1; W_m2[3, 3] = 1
    
    # M5: (A00+A01)B11
    U_m2[0, 4] = 1; U_m2[1, 4] = 1
    V_m2[3, 4] = 1
    W_m2[0, 4] = -1; W_m2[3, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U_m2[2, 5] = 1; U_m2[0, 5] = -1
    V_m2[0, 5] = 1; V_m2[1, 5] = 1
    W_m2[2, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U_m2[1, 6] = 1; U_m2[3, 6] = -1
    V_m2[2, 6] = 1; V_m2[3, 6] = 1
    W_m2[0, 6] = 1; W_m2[2, 6] = 1

    # 2. Build M4 Strassen Tensor (Rank 49)
    # A: 4x4. Blocks:
    # A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B: 4x4. Blocks similar.
    # C: 4x4. Blocks similar.
    
    # M4 has 7 "sub-tensors" S1..S7.
    # Each Sk is an M2 tensor applied to specific linear combos of blocks.
    
    # We will iterate through the 7 Strassen terms for the 4x4 level.
    # For each term, we determine the linear combination of A-blocks and B-blocks.
    # Then we expand using the M2 tensor.
    
    U_m4 = np.zeros((16, 49), dtype=int)
    V_m4 = np.zeros((16, 49), dtype=int)
    W_m4 = np.zeros((16, 49), dtype=int)
    
    # Helper to map block index and local index to global index
    # Block 0: rows 0-1, cols 0-1 (A00) -> Global indices: r*2+c
    # Block 1: rows 0-1, cols 2-3 (A01) -> Global indices: r*2+(c+2)
    # Block 2: rows 2-3, cols 0-1 (A10) -> Global indices: (r+2)*2+c
    # Block 3: rows 2-3, cols 2-3 (A11) -> Global indices: (r+2)*2+(c+2)
    
    def get_global_idx(block, r, c):
        # r, c are 0 or 1 (local 2x2 coords)
        base_r = (block // 2) * 2
        base_c = (block % 2) * 2
        return (base_r + r) * 2 + (base_c + c)
    
    # Strassen Terms for 4x4 (using blocks 0..3)
    # S1: (A00+A11)(B00+B11)
    # S2: (A10+A11)B00
    # S3: A00(B01-B11)
    # S4: A11(B10-B00)
    # S5: (A00+A01)B11
    # S6: (A10-A00)(B00+B01)
    # S7: (A01-A11)(B10+B11)
    
    # For each Sk, we have a linear map for A-blocks and B-blocks.
    # A_blocks: {0: coef0, 1: coef1, 2: coef2, 3: coef3}
    # B_blocks: {0: coef0, ...}
    
    terms = [
        ({0:1, 3:1}, {0:1, 3:1}), # S1
        ({2:1, 3:1}, {0:1}),      # S2
        ({0:1}, {1:1, 3:-1}),     # S3
        ({3:1}, {2:1, 0:-1}),     # S4
        ({0:1, 1:1}, {3:1}),      # S5
        ({2:1, 0:-1}, {0:1, 1:1}),# S6
        ({1:1, 3:-1}, {2:1, 3:1}) # S7
    ]
    
    col_m4 = 0
    for s_idx, (a_coefs, b_coefs) in enumerate(terms):
        # For each of the 7 columns in M2 tensor
        for r_m2 in range(7):
            # u_col = U_m2[:, r_m2] (length 4)
            # v_col = V_m2[:, r_m2] (length 4)
            # w_col = W_m2[:, r_m2] (length 4)
            
            # Expand U for this sub-term
            for a_blk, a_coef in a_coefs.items():
                if a_coef != 0:
                    # For each element in this block (4 elements)
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            global_idx = get_global_idx(a_blk, lr, lc)
                            # U_m2 contribution
                            u_val = U_m2[local_idx, r_m2]
                            if u_val != 0:
                                U_m4[global_idx, col_m4] += a_coef * u_val
                                
            # Expand V for this sub-term
            for b_blk, b_coef in b_coefs.items():
                if b_coef != 0:
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            global_idx = get_global_idx(b_blk, lr, lc)
                            v_val = V_m2[local_idx, r_m2]
                            if v_val != 0:
                                V_m4[global_idx, col_m4] += b_coef * v_val
                                
            # Expand W for this sub-term
            # W output blocks correspond to C blocks.
            # C blocks are 0..3 (C00, C01, C10, C11)
            # M2 W indices 0..3 correspond to C00, C01, C10, C11.
            for c_blk_idx in range(4):
                w_val = W_m2[c_blk_idx, r_m2]
                if w_val != 0:
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            # Output C block index is c_blk_idx
                            global_idx = get_global_idx(c_blk_idx, lr, lc)
                            W_m4[global_idx, col_m4] += w_val
                            
            col_m4 += 1
            
    # 3. Build Matrix-Vector Tensor (Rank 16)
    # A: 4x4, b: 4x1, c: 4x1
    # U: 16x16, V: 4x16, W: 1x4 (but mapped to global)
    
    U_mv = np.zeros((16, 16), dtype=int)
    V_mv = np.zeros((4, 16), dtype=int)
    W_mv = np.zeros((4, 16), dtype=int)
    
    for i in range(4):
        for j in range(4):
            col_mv = i * 4 + j
            U_mv[i * 4 + j, col_mv] = 1
            V_mv[j, col_mv] = 1
            W_mv[i, col_mv] = 1

    # 4. Combine
    # U_final: 16 x (49+16) = 16 x 65
    # V_final: 20 x 65
    #   Rows 0-15: V_m4 (4x4 part of B)
    #   Rows 16-19: V_mv (1x4 part of B, but V_mv is 4x16? No.
    #   V_mv has 4 rows. These correspond to the 4 elements of the 5th column of B.
    #   So V_final[16:20, 49:65] = V_mv
    # W_final: 20 x 65
    #   Rows 0-15: W_m4 (Output cols 0-3)
    #   Rows 16-19: W_mv (Output col 4)
    
    U_final = np.hstack([U_m4, U_mv])
    V_final = np.zeros((20, 65), dtype=int)
    V_final[0:16, 0:49] = V_m4
    V_final[16:20, 49:65] = V_mv
    
    W_final = np.zeros((20, 65), dtype=int)
    W_final[0:16, 0:49] = W_m4
    W_final[16:20, 49:65] = W_mv
    
    # Convert to list of lists as per return interface
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
