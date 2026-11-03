"""Rewritten program for 2x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's algorithm terms for 2x2x2
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Blocks: (a_col_offset, b_row_offset, b_col_offset, c_col_offset, base_rank)
    # A is 2x4, B is 4x4, C is 2x4
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    blocks = [
        (0, 0, 0, 0, 0),   # A_L * B_LL -> C_L
        (2, 2, 0, 0, 7),   # A_R * B_RL -> C_L
        (0, 0, 2, 2, 14),  # A_L * B_LR -> C_R
        (2, 2, 2, 2, 21),  # A_R * B_RR -> C_R
    ]

    for a_off, b_row_off, b_col_off, c_off, base_rank in blocks:
        for s in range(7):
            if time.time() > deadline:
                break
            col = base_rank + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # Process U (A matrix)
            # Global A index: i * 4 + j
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx_a = li * 4 + (a_off + lj)
                U[g_idx_a][col] += coeff
                
            # Process V (B matrix)
            # Global B index: k * 4 + r
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx_b = (b_row_off + li) * 4 + (b_col_off + lj)
                V[g_idx_b][col] += coeff
                
            # Process W (C matrix)
            # Global C index: r * 2 + i (Trace convention W[k*n+i, r] with n=2, k=r)
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx_c = (c_off + lj) * 2 + li
                W[g_idx_c][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
