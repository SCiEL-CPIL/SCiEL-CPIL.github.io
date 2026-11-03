import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    # We will use a rank-32 decomposition.
    # Split A (3x4) into A_top (2x4) and A_bottom (1x4).
    # Split B (4x4) into 2x2 blocks.
    # A_top * B uses 4 Strassen blocks (rank 28).
    # A_bottom * B uses 4 rank-1 terms (rank 4).
    # Total rank 32.
    
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen block for 2x2x2
    # U_loc: (4, 7), V_loc: (4, 7), W_loc: (4, 7)
    def strassen_block():
        U_loc = np.zeros((4, 7))
        V_loc = np.zeros((4, 7))
        W_loc = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        return U_loc, V_loc, W_loc
    
    U_loc, V_loc, W_loc = strassen_block()
    
    # Helper to add a Strassen block
    def add_strassen_block(col_offset, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # A is 2x2, B is 2x2, C is 2x2
        for r in range(2):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+7] += U_loc[local_a_idx, :]
                
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+7] += V_loc[local_b_idx, :]
                
        for r in range(2):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+7] += W_loc[local_c_idx, :]

    # Add 4 Strassen blocks for A_top (rows 0-1) * B
    # Block 0: A_top[:, 0:2] * B[0:2, 0:2] -> C_top[:, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A_top[:, 0:2] * B[0:2, 2:4] -> C_top[:, 2:4]
    add_strassen_block(7, 0, 0, 0, 2, 0, 2)
    
    # Block 2: A_top[:, 2:4] * B[2:4, 0:2] -> C_top[:, 0:2]
    add_strassen_block(14, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_top[:, 2:4] * B[2:4, 2:4] -> C_top[:, 2:4]
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Add 4 rank-1 terms for A_bottom (row 2) * B
    # C_bottom[:, k] = sum_j A_bottom[j] B[j, k]
    # Term r (0..3): A_bottom[r] * B_col_r
    # U[2*4 + r, 28+r] = 1
    # V[r*4 + k, 28+r] = 1 for all k? No.
    # V is (4x4, rank). V[j*4+k, r].
    # We need V[j*4+k, r] such that sum_j V[j*4+k, r] B[j*4+k] = B[r*4+k].
    # So V[r*4+k, r] = 1, others 0.
    # W[k*3 + 2, 28+r] = 1. (C index is k*3 + i, i=2)
    
    for r in range(4):
        col = 28 + r
        # U
        U[2 * 4 + r, col] = 1
        # V
        for k in range(4):
            V[r * 4 + k, col] = 1
        # W
        for k in range(4):
            W[k * 3 + 2, col] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
