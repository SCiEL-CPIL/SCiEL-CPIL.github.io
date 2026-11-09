import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define block indices for A, B, C
    # A: 4x4, blocks 2x2
    # Block 11: rows 0-1, cols 0-1
    # Block 12: rows 0-1, cols 2-3
    # Block 21: rows 2-3, cols 0-1
    # Block 22: rows 2-3, cols 2-3
    
    # Helper to get flattened indices for a block
    def get_block_indices(block_row, block_col):
        indices = []
        for i in range(2):
            for j in range(2):
                global_i = block_row * 2 + i
                global_j = block_col * 2 + j
                indices.append(global_i * m + global_j)
        return indices

    A_block_indices = {
        (0, 0): get_block_indices(0, 0),
        (0, 1): get_block_indices(0, 1),
        (1, 0): get_block_indices(1, 0),
        (1, 1): get_block_indices(1, 1)
    }
    
    B_block_indices = {
        (0, 0): get_block_indices(0, 0),
        (0, 1): get_block_indices(0, 1),
        (1, 0): get_block_indices(1, 0),
        (1, 1): get_block_indices(1, 1)
    }
    
    C_block_indices = {
        (0, 0): get_block_indices(0, 0),
        (0, 1): get_block_indices(0, 1),
        (1, 0): get_block_indices(1, 0),
        (1, 1): get_block_indices(1, 1)
    }
    
    # Level 1 Strassen Coefficients for A and B blocks
    # U_S = sum c_{S,b} A_b
    # V_S = sum d_{S,b} B_b
    # b is (br, bc) in {0,1}x{0,1}
    
    U_coeff = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    
    V_coeff = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (1,0): 1, (1,1): -1 },# M3
        { (0,0): 1, (0,1): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    
    # Level 2 Strassen Coefficients
    # u_T and v_T are 4-vectors corresponding to [u11, u12, u21, u22]
    u_strassen = [
        [1, 0, 0, 1], # T1
        [0, 0, 1, 1], # T2
        [1, 0, 0, 0], # T3
        [0, 0, 0, 1], # T4
        [1, 1, 0, 0], # T5
        [-1, 0, 1, 0],# T6
        [0, 1, 0, -1] # T7
    ]
    
    v_strassen = [
        [1, 0, 0, 1], # T1
        [1, 0, 0, 0], # T2
        [0, 0, 1, -1],# T3
        [1, -1, 0, 0],# T4
        [0, 0, 0, 1], # T5
        [1, 1, 0, 0], # T6
        [0, 0, 1, 1]  # T7
    ]
    
    # W coefficients for M_S entries from P_T
    # W[T] is 2x2 matrix
    W_strassen = [
        [[1, 0], [0, 1]], # T1
        [[0, 0], [1, 1]], # T2
        [[1, 0], [0, -1]],# T3
        [[-1, 0], [1, 0]],# T4
        [[0, 1], [0, -1]],# T5
        [[0, 0], [0, 0]], # T6 (Not used in output? Or maybe I missed it)
        [[1, 1], [0, 0]]  # T7? 
    ]
    # Correction for T7:
    # M11 = ... + M7
    # M12 = ... + M7
    # M21 = ...
    # M22 = ...
    # So W_T7 = [[1, 1], [0, 0]]
    
    # K coefficients: M_S contributes to C_blocks
    # K[S] is dict { (cr, cc): coeff }
    K_strassen = [
        { (0,0): 1, (1,1): 1 }, # M1 -> C11, C22
        { (1,0): 1, (1,1): 1 }, # M2 -> C21, C22
        { (0,0): 1, (1,1): -1 },# M3 -> C11, C22
        { (0,0): -1, (1,0): 1 },# M4 -> C11, C21
        { (0,1): 1, (1,1): -1 },# M5 -> C12, C22
        { (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0 }, # M6 -> 0
        { (0,0): 1, (0,1): 1 }  # M7 -> C11, C12
    ]
    
    col = 0
    for S in range(7):
        for T in range(7):
            # Construct U vector
            # U_S is 2x2 matrix of linear combos of A
            # u_T is coefficients for U_S entries
            # u_T[0]=u11, u_T[1]=u12, u_T[2]=u21, u_T[3]=u22
            
            u_vec = np.zeros(16, dtype=int)
            for r in range(2):
                for c in range(2):
                    coeff = u_strassen[T][r * 2 + c]
                    if coeff != 0:
                        # Add coeff * U_S[r,c]
                        # U_S[r,c] is sum of blocks
                        for b, cb in U_coeff[S].items():
                            if cb != 0:
                                # A_b indices
                                for idx in A_block_indices[b]:
                                    # Note: U_S[r,c] uses the (r,c) entry of the block A_b
                                    # The flattened index idx corresponds to A[i,j]
                                    # We need to map (b, r, c) to global (i, j)
                                    # b is (br, bc). i = br*2+r, j = bc*2+c
                                    # This is exactly what get_block_indices does?
                                    # get_block_indices(br, bc) returns indices for all r,c in that block.
                                    # We need to pick the specific index for r,c.
                                    # The indices in A_block_indices[b] are [i0j0, i0j1, i1j0, i1j1]
                                    # r=0,c=0 -> idx 0
                                    # r=0,c=1 -> idx 1
                                    # r=1,c=0 -> idx 2
                                    # r=1,c=1 -> idx 3
                                    local_idx = r * 2 + c
                                    global_idx = A_block_indices[b][local_idx]
                                    u_vec[global_idx] += coeff * cb
                                    
            U[:, col] = u_vec
            
            # Construct V vector
            v_vec = np.zeros(16, dtype=int)
            for r in range(2):
                for c in range(2):
                    coeff = v_strassen[T][r * 2 + c]
                    if coeff != 0:
                        for b, cb in V_coeff[S].items():
                            if cb != 0:
                                local_idx = r * 2 + c
                                global_idx = B_block_indices[b][local_idx]
                                v_vec[global_idx] += coeff * cb
                                
            V[:, col] = v_vec
            
            # Construct W vector
            w_vec = np.zeros(16, dtype=int)
            # W_strassen[T] is 2x2 matrix W_T
            # M_S entry (r,c) gets coeff W_T[r,c]
            # M_S contributes to C_blocks with K_strassen[S]
            
            for cr, cc in K_strassen[S].items():
                coeff_K = K_strassen[S][(cr, cc)]
                if coeff_K == 0:
                    continue
                # For each entry (r,c) in M_S
                for r in range(2):
                    for c in range(2):
                        coeff_W = W_strassen[T][r][c]
                        if coeff_W == 0:
                            continue
                        total_coeff = coeff_K * coeff_W
                        if total_coeff != 0:
                            # This contributes to C_block (cr, cc) at position (r,c)
                            # Global C index:
                            # i = cr*2 + r
                            # k = cc*2 + c
                            # W index: k*n + i
                            i = cr * 2 + r
                            k = cc * 2 + c
                            w_idx = k * n + i
                            w_vec[w_idx] += total_coeff
                            
            W[:, col] = w_vec
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
