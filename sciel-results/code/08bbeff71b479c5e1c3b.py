"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    def fallback_schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen 2x2x2 coefficients
    strassen_coeffs = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Outer Strassen for 4x4x4
    outer_coeffs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0,0): 1, (1,1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1,0): 1, (1,1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0,1): 1, (1,1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0,0): 1, (1,0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0,0): -1, (0,1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1,1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0,0): 1})
    ]

    total_rank = 93
    mat_u = np.zeros((n * m, total_rank))
    mat_v = np.zeros((m * p, total_rank))
    mat_w = np.zeros((p * n, total_rank))
    
    curr_col = 0

    # Part 1: 4x4x4 (cols 0-3) using Outer Strassen (Rank 49)
    for o_idx in range(7):
        a_blocks, b_blocks, c_block_map = outer_coeffs[o_idx]
        for s_idx in range(7):
            u_dict, v_dict, w_dict = strassen_coeffs[s_idx]
            r = curr_col
            curr_col += 1
            
            # Fill U
            for (li, lj), coeff_u in u_dict.items():
                for b_id, coeff_b in a_blocks.items():
                    bi, bj = b_id // 2, b_id % 2
                    i_glob = 2 * bi + li
                    j_glob = 2 * bj + lj
                    mat_u[i_glob * m + j_glob, r] += coeff_u * coeff_b

            # Fill V
            for (li, lj), coeff_v in v_dict.items():
                for b_id, coeff_b in b_blocks.items():
                    bi, bj = b_id // 2, b_id % 2
                    i_glob = 2 * bi + li
                    j_glob = 2 * bj + lj
                    mat_v[i_glob * p + j_glob, r] += coeff_v * coeff_b

            # Fill W
            for (li, lj), coeff_w in w_dict.items():
                if coeff_w == 0: continue
                for (br, bc), coeff_c in c_block_map.items():
                    if coeff_c == 0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    mat_w[k_final * n + i_final, r] += coeff_w * coeff_c

    # Part 2: 4x4x2 (cols 4-5) using 4x Strassen 2x2x2 (Rank 28)
    def append_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal curr_col
        for s in strassen_coeffs:
            u_d, v_d, w_d = s
            r = curr_col
            curr_col += 1
            for (li, lj), val in u_d.items():
                i = a_rows[li]
                j = a_cols[lj]
                mat_u[i * m + j, r] += val
            for (li, lj), val in v_d.items():
                i = b_rows[li]
                j = b_cols[lj]
                mat_v[i * p + j, r] += val
            for (li, lj), val in w_d.items():
                i = c_rows[li]
                k = c_cols[lj]
                mat_w[k * n + i, r] += val

    # Block definitions for Part 2
    A00_r = [0, 1]; A00_c = [0, 1]
    A01_r = [0, 1]; A01_c = [2, 3]
    A10_r = [2, 3]; A10_c = [0, 1]
    A11_r = [2, 3]; A11_c = [2, 3]
    
    B_top_r = [0, 1]; B_top_c = [4, 5]
    B_bot_r = [2, 3]; B_bot_c = [4, 5]
    
    C_top_r = [0, 1]; C_top_c = [4, 5]
    C_bot_r = [2, 3]; C_bot_c = [4, 5]

    append_strassen_block(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
    append_strassen_block(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
    append_strassen_block(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
    append_strassen_block(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)

    # Part 3: 4x4x1 (col 6) using Schoolbook (Rank 16)
    for i in range(n):
        for j in range(m):
            r = curr_col
            curr_col += 1
            mat_u[i * m + j, r] = 1.0
            mat_v[j * p + 6, r] = 1.0
            mat_w[6 * n + i, r] = 1.0

    if curr_col != total_rank:
        return fallback_schoolbook()

    # Check deadline before expensive verification
    if time.time() > deadline - 5.0:
        return {"U": mat_u.tolist(), "V": mat_v.tolist(), "W": mat_w.tolist()}

    U_list = mat_u.tolist()
    V_list = mat_v.tolist()
    W_list = mat_w.tolist()

    # Verification using vectorized computation
    Ua = np.array(U_list)
    Va = np.array(V_list)
    Wa = np.array(W_list)
    
    # Pre-compute index arrays for verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    calc = (Ua[u_idx] * Va[v_idx] * Wa[w_idx]).sum(axis=1)
    
    if np.max(np.abs(calc - 1.0)) > 1e-8:
        return fallback_schoolbook()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
