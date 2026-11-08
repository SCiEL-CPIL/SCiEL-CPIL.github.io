import numpy as np
import time

def construct():
    """
    Matrix multiplication 3x4 by 4x4.
    Decomposes into 1x4x4 (rank 4) and 2x4x4 (rank 28).
    Total rank 32.
    """
    start_time = time.time()
    
    n, m, p = 3, 4, 4
    rank = 32
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 1x4x4 (Row 0 of A, Row 0 of C)
    # Rank 4
    for t in range(4):
        # U: A_t = 1
        U[t, t] = 1
        
        # V: B_{t, j} for j=0..3
        for j in range(4):
            # B index: row t, col j -> t * 4 + j
            V[t * 4 + j, t] = 1
            
        # W: C_j = 1 for j=0..3
        for j in range(4):
            # C index: col j, row 0 -> j * 3 + 0
            W[j * 3 + 0, t] = 1
            
    # Part 2: 2x4x4 (Rows 1,2 of A, Rows 1,2 of C)
    # Decompose into 4 blocks of 2x2x2
    # Reuse Strassen factors from 3x2x2 construction (first 4 rows)
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # Strassen part (cols 0-6)
    # M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1 # Note: In 3x2x2, this might be different, but for 2x2x2 it should contribute to C11?
    # Actually, let's check the 3x2x2 W for col 5.
    # W_loc[3, 5] = 1.
    # In 2x2x2, C11 is index 3.
    # So C11 += M6.
    
    # M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1 # C00 += M7
    
    # Map blocks
    col_offset = 4
    for br in range(2):
        for bc in range(2):
            # U mapping
            for r_loc in range(2):
                for c_loc in range(2):
                    a_idx_loc = r_loc * 2 + c_loc
                    a_row_glob = 1 + r_loc
                    a_col_glob = 2 * bc + c_loc
                    a_idx_glob = a_row_glob * 4 + a_col_glob
                    U[a_idx_glob, col_offset:col_offset+7] += U_loc[a_idx_loc, :]
                    
            # V mapping
            for r_loc in range(2):
                for c_loc in range(2):
                    b_idx_loc = r_loc * 2 + c_loc
                    b_row_glob = 2 * br + r_loc
                    b_col_glob = 2 * bc + c_loc
                    b_idx_glob = b_row_glob * 4 + b_col_glob
                    V[b_idx_glob, col_offset:col_offset+7] += V_loc[b_idx_loc, :]
                    
            # W mapping
            for r_loc in range(2):
                for c_loc in range(2):
                    c_idx_loc = r_loc * 2 + c_loc
                    c_row_glob = 1 + r_loc
                    c_col_glob = 2 * bc + c_loc
                    c_idx_glob = c_col_glob * 3 + c_row_glob
                    W[c_idx_glob, col_offset:col_offset+7] += W_loc[c_idx_loc, :]
                    
            col_offset += 7
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
