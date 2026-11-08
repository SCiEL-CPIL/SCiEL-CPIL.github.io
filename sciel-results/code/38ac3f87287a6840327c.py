import numpy as np
import time

def construct():
    """
    Matrix multiplication 3x4 by 4x4.
    Uses the known optimal rank 44 construction derived from
    splitting the 4x4 block into four 2x2 blocks and applying
    an 11-rank decomposition for 3x2 * 2x2 -> 3x2.
    Total rank = 4 * 11 = 44.
    """
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    # Local A indices: 0..5 (row * 2 + col)
    # Local B indices: 0..3 (row * 2 + col)
    # Local C indices: 0..5 (row * 2 + col)
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    # M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1 # c00, c11
    
    # M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1 # c10, c11
    
    # M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1 # c01, c11
    
    # M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1 # c00, c10
    
    # M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1 # c00, c01
    
    # M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1 # c11
    
    # M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1 # c00
    
    # Extra part (cols 7-10) for Row 2 of A
    # M8 = a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9 = a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # M10 = a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11 = a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Mapping function
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3): # Local A rows
            for c in range(2): # Local A cols
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                
                # Copy U_loc
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
                
        for r in range(2): # Local B rows
            for c in range(2): # Local B cols
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                
                # Copy V_loc
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
                
        for r in range(3): # Local C rows
            for c in range(2): # Local C cols
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                
                # Copy W_loc
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    # Block 0: A_L (cols 0,1) * B_11 (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A_R (cols 2,3) * B_21 (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    add_block(1, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A_L (cols 0,1) * B_12 (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    add_block(2, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A_R (cols 2,3) * B_22 (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
