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

    # Rank 1 part: A1 (2x1) B1 (1x6)
    # j=0
    U_part1 = np.zeros((nm, 1))
    V_part1 = np.zeros((mp, 1))
    W_part1 = np.zeros((pn, 1))
    
    # A1 indices: i in 0..1, j=0
    for i in range(n):
        U_part1[i * m + 0, 0] = 1.0
        
    # B1 indices: j=0, k in 0..5
    for k in range(p):
        V_part1[0 * p + k, 0] = 1.0
        
    # Output indices: k in 0..5, i in 0..1
    for k in range(p):
        for i in range(n):
            W_part1[k * n + i, 0] = 1.0

    # Helper to add Strassen 2x2 by 2x2 decomposition
    # a_offset: index offset in U for A (i*m + j)
    # b_offset: index offset in V for B (j*p + k)
    # c_offset: index offset in W for C (k*n + i)
    # j_start: starting j index for A and B
    # k_start: starting k index for B
    # This adds 7 columns to U, V, W
    
    def add_strassen_2x2_2x2(U, V, W, j_start, k_start, col_offset):
        # A is 2x2: i in 0..1, j in j_start..j_start+1
        # B is 2x2: j in j_start..j_start+1, k in k_start..k_start+1
        # C is 2x2: k in k_start..k_start+1, i in 0..1
        
        # Strassen matrices for 2x2
        # U_A (4x7), V_B (4x7), W_C (4x7)
        # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U_A columns
        u_a = np.zeros((4, 7))
        u_a[0, 0] = 1; u_a[3, 0] = 1
        u_a[2, 1] = 1; u_a[3, 1] = 1
        u_a[0, 2] = 1
        u_a[3, 3] = 1
        u_a[0, 4] = 1; u_a[1, 4] = 1
        u_a[2, 5] = 1; u_a[0, 5] = -1
        u_a[1, 6] = 1; u_a[3, 6] = -1
        
        # V_B columns
        v_b = np.zeros((4, 7))
        v_b[0, 0] = 1; v_b[3, 0] = 1
        v_b[0, 1] = 1
        v_b[2, 2] = 1; v_b[3, 2] = -1
        v_b[1, 3] = 1; v_b[0, 3] = -1
        v_b[3, 4] = 1
        v_b[0, 5] = 1; v_b[1, 5] = 1
        v_b[2, 6] = 1; v_b[3, 6] = 1
        
        # W_C columns
        w_c = np.zeros((4, 7))
        w_c[0, 0] = 1; w_c[1, 0] = 1; w_c[2, 0] = 1; w_c[3, 0] = 1
        w_c[0, 1] = 1; w_c[3, 1] = 1
        w_c[2, 2] = 1; w_c[1, 2] = 1
        w_c[1, 3] = 1; w_c[2, 3] = 1
        w_c[1, 4] = 1; w_c[3, 4] = 1
        w_c[0, 5] = 1; w_c[3, 5] = 1
        w_c[2, 6] = 1; w_c[1, 6] = 1

        # Map local indices to global
        # A local (li, lj) -> global i, j = j_start + lj
        # B local (lj, lk) -> global j = j_start + lj, k = k_start + lk
        # C local (lk, li) -> global k = k_start + lk, i
        
        for r in range(7):
            col = col_offset + r
            # U
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_i = li
                    global_j = j_start + lj
                    U[global_i * m + global_j, col] += u_a[local_idx, r]
                    
            # V
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    global_j = j_start + lj
                    global_k = k_start + lk
                    V[global_j * p + global_k, col] += v_b[local_idx, r]
                    
            # W
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_k = k_start + lk
                    global_i = li
                    W[global_k * n + global_i, col] += w_c[local_idx, r]

    # Initialize U, V, W
    total_rank = 1 + 21 + 21
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Copy Part 1
    U[:, 0] = U_part1[:, 0]
    V[:, 0] = V_part1[:, 0]
    W[:, 0] = W_part1[:, 0]
    
    # Part 21: A21 (j=1,2) B21 (j=1,2; k=0..5)
    # Split B21 into 3 blocks of 2x2: k=0,1; k=2,3; k=4,5
    col_off = 1
    for k_block in range(3):
        k_start = k_block * 2
        add_strassen_2x2_2x2(U, V, W, 1, k_start, col_off)
        col_off += 7
        
    # Part 22: A22 (j=3,4) B22 (j=3,4; k=0..5)
    # Split B22 into 3 blocks of 2x2: k=0,1; k=2,3; k=4,5
    for k_block in range(3):
        k_start = k_block * 2
        add_strassen_2x2_2x2(U, V, W, 3, k_start, col_off)
        col_off += 7

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
