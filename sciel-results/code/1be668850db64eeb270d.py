import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    total_rank = 88
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # i_off: global row offset for A (0 or 1)
        # j_off: global col offset for A / row offset for B (0 or 1)
        # k_off: global col offset for B / row offset for C (0 or 1)
        
        # Strassen vectors for 2x2x2
        # U indices: i*2 + j
        # V indices: j*2 + k
        # W indices: k*2 + i
        
        U_vecs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1]  # M7
        ]
        V_vecs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1], # M3
            [-1, 0, 1, 0], # M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        W_vecs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1], # M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0], # M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        for t in range(7):
            r = col_start + t
            # Fill U
            for local_i in range(2):
                for local_j in range(2):
                    val = U_vecs[t][local_i * 2 + local_j]
                    if val != 0:
                        global_i = i_off + local_i
                        global_j = j_off + local_j
                        U[global_i * m + global_j, r] = val
            # Fill V
            for local_j in range(2):
                for local_k in range(2):
                    val = V_vecs[t][local_j * 2 + local_k]
                    if val != 0:
                        global_j = j_off + local_j
                        global_k = k_off + local_k
                        V[global_j * p + global_k, r] = val
            # Fill W
            for local_k in range(2):
                for local_i in range(2):
                    val = W_vecs[t][local_k * 2 + local_i]
                    if val != 0:
                        global_k = k_off + local_k
                        global_i = i_off + local_i
                        W[global_k * n + global_i, r] = val
                        
    # Part 1: Row 0 (1x4x8) - Naive
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: Rows 1-2, Cols 0-3 (2x4x4)
    # A sub-blocks: A[1:3, 0:2] (i_off=1, j_off=0), A[1:3, 2:4] (i_off=1, j_off=2)
    # B sub-blocks: B[0:2, 0:2] (j_off=0, k_off=0), B[0:2, 2:4] (j_off=0, k_off=2)
    #               B[2:4, 0:2] (j_off=2, k_off=0), B[2:4, 2:4] (j_off=2, k_off=2)
    
    # C_L = A_L B_LL + A_R B_RL
    # A_L is at i_off=1, j_off=0. B_LL is at j_off=0, k_off=0.
    add_strassen_block(1, 0, 0, col)
    col += 7
    
    # A_R is at i_off=1, j_off=2. B_RL is at j_off=2, k_off=0.
    add_strassen_block(1, 2, 0, col)
    col += 7
    
    # C_R = A_L B_LR + A_R B_RR
    # A_L is at i_off=1, j_off=0. B_LR is at j_off=0, k_off=2.
    add_strassen_block(1, 0, 2, col)
    col += 7
    
    # A_R is at i_off=1, j_off=2. B_RR is at j_off=2, k_off=2.
    add_strassen_block(1, 2, 2, col)
    col += 7
    
    # Part 3: Rows 1-2, Cols 4-7 (2x4x4)
    # B sub-blocks for right half:
    # B[0:2, 4:6] (j_off=0, k_off=4), B[0:2, 6:8] (j_off=0, k_off=6)
    # B[2:4, 4:6] (j_off=2, k_off=4), B[2:4, 6:8] (j_off=2, k_off=6)
    
    # A_L (i_off=1, j_off=0) * B_LL_right (j_off=0, k_off=4)
    add_strassen_block(1, 0, 4, col)
    col += 7
    
    # A_R (i_off=1, j_off=2) * B_RL_right (j_off=2, k_off=4)
    add_strassen_block(1, 2, 4, col)
    col += 7
    
    # A_L (i_off=1, j_off=0) * B_LR_right (j_off=0, k_off=6)
    add_strassen_block(1, 0, 6, col)
    col += 7
    
    # A_R (i_off=1, j_off=2) * B_RR_right (j_off=2, k_off=6)
    add_strassen_block(1, 2, 6, col)
    col += 7
    
    return {"U": U, "V": V, "W": W}
