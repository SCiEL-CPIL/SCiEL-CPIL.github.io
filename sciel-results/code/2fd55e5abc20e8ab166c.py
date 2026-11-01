import numpy as np

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a block
    # u_block: (4, r)
    # v_block: (4, r) -- local B indices
    # w_block: (4, r) -- local C indices
    # v_offset: list of 4 global V row indices corresponding to local B 0,1,2,3
    # w_offset: list of 4 global W row indices corresponding to local C 0,1,2,3
    # start_col: starting column in U,V,W
    
    def add_block(u_block, v_block, w_block, v_map, w_map, start_col):
        r = u_block.shape[1]
        U[:, start_col:start_col+r] = u_block
        for local_j in range(4):
            global_row = v_map[local_j]
            if global_row is not None:
                V[global_row, start_col:start_col+r] = v_block[local_j, :]
        for local_k in range(4):
            global_row = w_map[local_k]
            if global_row is not None:
                W[global_row, start_col:start_col+r] = w_block[local_k, :]
        return start_col + r

    # Strassen Block (2x2x2)
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 1]
    ])
    
    # Schoolbook Block (2x2x1)
    U_SB = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_SB = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    # Note: V_SB needs to be 4x4 for the helper, but only 2 rows are used.
    # We can just pass a 4x4 with zeros.
    V_SB_full = np.zeros((4, 4))
    V_SB_full[0, :] = V_SB[0, :]
    V_SB_full[1, :] = V_SB[1, :]
    
    W_SB = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    W_SB_full = np.zeros((4, 4))
    W_SB_full[0, :] = W_SB[0, :]
    W_SB_full[1, :] = W_SB[1, :]

    col = 0
    
    # Block 1: Cols 0, 1 (Strassen)
    # V Map: Local B 0,1,2,3 -> Global B
    # B index j*p + k. p=7.
    # Local 0 (0,0) -> 0*7+0 = 0
    # Local 1 (0,1) -> 0*7+1 = 1
    # Local 2 (1,0) -> 1*7+0 = 7
    # Local 3 (1,1) -> 1*7+1 = 8
    v_map_1 = [0, 1, 7, 8]
    
    # W Map: Local C 0,1,2,3 -> Global C
    # C index k*n + i. n=2.
    # Local 0 (0,0) -> 0*2+0 = 0
    # Local 1 (1,0) -> 0*2+1 = 1
    # Local 2 (0,1) -> 1*2+0 = 2
    # Local 3 (1,1) -> 1*2+1 = 3
    w_map_1 = [0, 1, 2, 3]
    
    col = add_block(U_S, V_S, W_S, v_map_1, w_map_1, col)
    
    # Block 2: Cols 2, 3 (Strassen)
    # V Map:
    # Local 0 (0,0) -> 0*7+2 = 2
    # Local 1 (0,1) -> 0*7+3 = 3
    # Local 2 (1,0) -> 1*7+2 = 9
    # Local 3 (1,1) -> 1*7+3 = 10
    v_map_2 = [2, 3, 9, 10]
    
    # W Map:
    # Local 0 (0,0) -> 2*2+0 = 4
    # Local 1 (1,0) -> 2*2+1 = 5
    # Local 2 (0,1) -> 3*2+0 = 6
    # Local 3 (1,1) -> 3*2+1 = 7
    w_map_2 = [4, 5, 6, 7]
    
    col = add_block(U_S, V_S, W_S, v_map_2, w_map_2, col)
    
    # Block 3: Cols 4, 5, 6 (Strassen for 4,5 + Schoolbook for 6)
    # This is a 2x2x3 block.
    # We can treat it as Strassen (cols 4,5) + Schoolbook (col 6).
    
    # Part 3a: Strassen for cols 4,5
    v_map_3a = [4, 5, 11, 12] # B 0,4; 0,5; 1,4; 1,5
    w_map_3a = [8, 9, 10, 11] # C 4,0; 4,1; 5,0; 5,1
    
    col = add_block(U_S, V_S, W_S, v_map_3a, w_map_3a, col)
    
    # Part 3b: Schoolbook for col 6
    # V Map:
    # Local 0 (0,0) -> 0*7+6 = 6
    # Local 1 (1,0) -> 1*7+6 = 13
    # Local 2,3 unused (None)
    v_map_3b = [6, 13, None, None]
    
    # W Map:
    # Local 0 (0,0) -> 6*2+0 = 12
    # Local 1 (1,0) -> 6*2+1 = 13
    # Local 2,3 unused
    w_map_3b = [12, 13, None, None]
    
    col = add_block(U_SB, V_SB_full, W_SB_full, v_map_3b, w_map_3b, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
