import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Local 3x2x2 decomposition of rank 11
    # U_loc: 6x11, V_loc: 4x11, W_loc: 6x11
    
    # Strassen 2x2x2 part (rank 7)
    # m1..m7
    U_strassen = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a00
        [0, 0, 0, 1, 1, 1, 1], # a01
        [0, 1, -1, 0, 0, -1, 0], # a10
        [1, 1, 0, 1, 0, 0, 1]  # a11
    ])
    V_strassen = np.array([
        [0, 1, 1, 0, 0, 1, 0], # b00
        [1, 0, 0, 1, 0, 0, 1], # b01
        [1, 0, -1, 0, 0, 1, 0], # b10
        [0, 0, 0, -1, 1, 0, 1]  # b11
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, -1], # c00
        [0, 0, 1, 0, 0, 0, 1],   # c01
        [0, 1, 0, 0, 0, 1, 0],   # c10
        [1, -1, 1, 0, 1, 0, 0]   # c11
    ])
    
    # Extend to 3x2x2 with 4 more multiplications
    # m8 = a20 b00
    # m9 = a21 b10
    # m10 = a20 b01
    # m11 = a21 b11
    
    U_ext = np.zeros((6, 4))
    U_ext[4, 0] = 1 # a20 for m8
    U_ext[5, 1] = 1 # a21 for m9
    U_ext[4, 2] = 1 # a20 for m10
    U_ext[5, 3] = 1 # a21 for m11
    
    V_ext = np.zeros((4, 4))
    V_ext[0, 0] = 1 # b00 for m8
    V_ext[2, 1] = 1 # b10 for m9
    V_ext[1, 2] = 1 # b01 for m10
    V_ext[3, 3] = 1 # b11 for m11
    
    W_ext = np.zeros((6, 4))
    # c20 = m8 + m9 -> idx 4
    W_ext[4, 0] = 1
    W_ext[4, 1] = 1
    # c21 = m10 + m11 -> idx 5
    W_ext[5, 2] = 1
    W_ext[5, 3] = 1
    
    U_loc = np.hstack([U_strassen, U_ext])
    V_loc = np.hstack([V_strassen, V_ext])
    W_loc = np.hstack([W_strassen, W_ext])
    
    # Global dimensions
    rank = 44
    U = np.zeros((12, rank))
    V = np.zeros((16, rank))
    W = np.zeros((12, rank))
    
    # Blocks:
    # 1: A cols 0,1; B rows 0,1; C cols 0,1
    # 2: A cols 0,1; B rows 0,1; C cols 2,3
    # 3: A cols 2,3; B rows 2,3; C cols 0,1
    # 4: A cols 2,3; B rows 2,3; C cols 2,3
    
    blocks = [
        # (a_col_start, b_row_start, c_col_start)
        (0, 0, 0),
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2)
    ]
    
    for b_idx, (ac, br, cc) in enumerate(blocks):
        col_offset = b_idx * 11
        
        # U rows: i in 0..2, j in ac..ac+1
        for i in range(3):
            for j in range(2):
                u_row = i * 4 + (ac + j)
                u_loc_row = i * 2 + j
                U[u_row, col_offset:col_offset+11] = U_loc[u_loc_row, :]
                
        # V rows: j in br..br+1, k in 0..1 (local k is 0..1)
        # Wait, V local index is j_loc * 2 + k_loc
        # Global V index is j_global * 4 + k_global
        for j_loc in range(2):
            for k_loc in range(2):
                v_row = (br + j_loc) * 4 + (cc + k_loc)
                v_loc_row = j_loc * 2 + k_loc
                V[v_row, col_offset:col_offset+11] = V_loc[v_loc_row, :]
                
        # W rows: k in cc..cc+1, i in 0..2
        # Global W index is k_global * 3 + i
        for k_loc in range(2):
            for i in range(3):
                w_row = (cc + k_loc) * 3 + i
                w_loc_row = k_loc * 3 + i
                W[w_row, col_offset:col_offset+11] = W_loc[w_loc_row, :]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
