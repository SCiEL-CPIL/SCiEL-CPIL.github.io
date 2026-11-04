import sys
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                T[idx1, idx2, idx3] = 1.0

    def verify(U, V, W):
        rank = U.shape[1]
        T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-8)

    # Helper to add Strassen 2x2 by 2x2 decomposition
    def add_strassen_2x2_2x2(U, V, W, j_start, k_start, col_offset):
        # A is 2x2: i in 0..1, j in j_start..j_start+1
        # B is 2x2: j in j_start..j_start+1, k in k_start..k_start+1
        # C is 2x2: k in k_start..k_start+1, i in 0..1
        
        u_a = np.zeros((4, 7))
        u_a[0, 0] = 1; u_a[3, 0] = 1
        u_a[2, 1] = 1; u_a[3, 1] = 1
        u_a[0, 2] = 1
        u_a[3, 3] = 1
        u_a[0, 4] = 1; u_a[1, 4] = 1
        u_a[2, 5] = 1; u_a[0, 5] = -1
        u_a[1, 6] = 1; u_a[3, 6] = -1
        
        v_b = np.zeros((4, 7))
        v_b[0, 0] = 1; v_b[3, 0] = 1
        v_b[0, 1] = 1
        v_b[2, 2] = 1; v_b[3, 2] = -1
        v_b[1, 3] = 1; v_b[0, 3] = -1
        v_b[3, 4] = 1
        v_b[0, 5] = 1; v_b[1, 5] = 1
        v_b[2, 6] = 1; v_b[3, 6] = 1
        
        w_c = np.zeros((4, 7))
        w_c[0, 0] = 1; w_c[1, 0] = 1; w_c[2, 0] = 1; w_c[3, 0] = 1
        w_c[0, 1] = 1; w_c[3, 1] = 1
        w_c[2, 2] = 1; w_c[1, 2] = 1
        w_c[1, 3] = 1; w_c[2, 3] = 1
        w_c[1, 4] = 1; w_c[3, 4] = 1
        w_c[0, 5] = 1; w_c[3, 5] = 1
        w_c[2, 6] = 1; w_c[1, 6] = 1

        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_i = li
                    global_j = j_start + lj
                    U[global_i * m + global_j, col] += u_a[local_idx, r]
                    
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    global_j = j_start + lj
                    global_k = k_start + lk
                    V[global_j * p + global_k, col] += v_b[local_idx, r]
                    
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_k = k_start + lk
                    global_i = li
                    W[global_k * n + global_i, col] += w_c[local_idx, r]

    # Helper to add 2x2 by 2x1 decomposition (Rank 2)
    def add_2x2_2x1(U, V, W, j_start, k_start, col_offset):
        # A is 2x2: i in 0..1, j in j_start..j_start+1
        # B is 2x1: j in j_start..j_start+1, k = k_start
        # C is 2x1: k = k_start, i in 0..1
        
        # r=0: A_i0 B_00
        col = col_offset
        for i in range(2):
            U[i * m + j_start, col] += 1.0
        V[j_start * p + k_start, col] += 1.0
        for i in range(2):
            W[k_start * n + i, col] += 1.0
            
        # r=1: A_i1 B_10
        col = col_offset + 1
        for i in range(2):
            U[i * m + (j_start + 1), col] += 1.0
        V[(j_start + 1) * p + k_start, col] += 1.0
        for i in range(2):
            W[k_start * n + i, col] += 1.0

    # Helper to add 2x1 by 1x3 decomposition (Rank 3)
    def add_2x1_1x3(U, V, W, j_start, k_start, col_offset):
        # A is 2x1: i in 0..1, j = j_start
        # B is 1x3: j = j_start, k in k_start..k_start+2
        # C is 2x3: k in k_start..k_start+2, i in 0..1
        
        for r in range(3):
            col = col_offset + r
            k_idx = k_start + r
            for i in range(2):
                U[i * m + j_start, col] += 1.0
            V[j_start * p + k_idx, col] += 1.0
            for i in range(2):
                W[k_idx * n + i, col] += 1.0

    # Helper to add 2x3 by 3x3 decomposition (Rank 12)
    def add_2x3_3x3(U, V, W, j_start, k_start, col_offset):
        # A is 2x3: i in 0..1, j in j_start..j_start+2
        # B is 3x3: j in j_start..j_start+2, k in k_start..k_start+2
        # C is 2x3: k in k_start..k_start+2, i in 0..1
        
        # Split A into A1 (2x2) and A2 (2x1)
        # Split B into B1 (2x3) and B2 (1x3)
        
        # Part 1: A1 (2x2) * B1 (2x3)
        # B1 is 2x3, split into B1a (2x2) and B1b (2x1)
        
        # A1 * B1a (2x2 * 2x2) -> Rank 7
        add_strassen_2x2_2x2(U, V, W, j_start, k_start, col_offset)
        col_offset += 7
        
        # A1 * B1b (2x2 * 2x1) -> Rank 2
        # j_start is same, k_start+2 is the new k
        add_2x2_2x1(U, V, W, j_start, k_start + 2, col_offset)
        col_offset += 2
        
        # Part 2: A2 (2x1) * B2 (1x3)
        # A2 is at j_start+2
        # B2 is at j_start+2, k in k_start..k_start+2
        add_2x1_1x3(U, V, W, j_start + 2, k_start, col_offset)
        col_offset += 3
        
        return col_offset

    # Initialize U, V, W
    # Rank 45
    total_rank = 45
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col_off = 0
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # B1 is 2x6, split into three 2x2 blocks
    for k_block in range(3):
        k_start = k_block * 2
        add_strassen_2x2_2x2(U, V, W, 0, k_start, col_off)
        col_off += 7
        
    # Part 2: A2 (2x3) * B2 (3x6)
    # B2 is 3x6, split into two 3x3 blocks
    for k_block in range(2):
        k_start = k_block * 3
        col_off = add_2x3_3x3(U, V, W, 2, k_start, col_off)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U0 = np.zeros((nm, rank))
        V0 = np.zeros((mp, rank))
        W0 = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i * m + j
                    idx2 = j * p + k
                    idx3 = k * n + i
                    U0[idx1, col] = 1.0
                    V0[idx2, col] = 1.0
                    W0[idx3, col] = 1.0
                    col += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

# EVOLVE-BLOCK-END
