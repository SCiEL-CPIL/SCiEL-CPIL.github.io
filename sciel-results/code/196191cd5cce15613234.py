import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 150s limit, 5s margin
    
    n, m, p = 4, 4, 5
    rank = 49 + 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4x4 (Strassen) ---
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # p1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # p2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # p3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # p4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # p5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # p6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # p7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    blocks = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, 1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]
    
    def get_block_indices(block_idx):
        if block_idx == 0: return [0, 1, 4, 5]
        if block_idx == 1: return [2, 3, 6, 7]
        if block_idx == 2: return [8, 9, 12, 13]
        if block_idx == 3: return [10, 11, 14, 15]
        return []
        
    for k in range(7):
        if time.time() > deadline:
            break
        block_def = blocks[k]
        for l in range(7):
            col = k * 7 + l
            
            for r in range(4):
                u_val = U_loc[r, l]
                if u_val != 0:
                    for b_idx, coeff in block_def['A']:
                        g_inds = get_block_indices(b_idx)
                        i_loc = r // 2
                        j_loc = r % 2
                        g_ind = g_inds[2 * i_loc + j_loc]
                        U[g_ind, col] += u_val * coeff
                        
            for r in range(4):
                v_val = V_loc[r, l]
                if v_val != 0:
                    for b_idx, coeff in block_def['B']:
                        g_inds = get_block_indices(b_idx)
                        i_loc = r // 2
                        j_loc = r % 2
                        g_ind = g_inds[2 * i_loc + j_loc]
                        V[g_ind, col] += v_val * coeff
                        
            for r in range(4):
                w_val = W_loc[r, l]
                if w_val != 0:
                    for b_idx, coeff in block_def['C']:
                        g_inds = get_block_indices(b_idx)
                        i_loc = r // 2
                        j_loc = r % 2
                        g_ind = g_inds[2 * i_loc + j_loc]
                        W[g_ind, col] += w_val * coeff

    # --- Part 2: 4x4x1 (Naive) ---
    col_offset = 49
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
