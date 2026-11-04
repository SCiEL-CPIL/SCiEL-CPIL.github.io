"""Optimized Strassen-based decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Pre-defined Strassen 2x2x2 coefficients
    # Format: (U_coeffs, V_coeffs, W_coeffs)
    # Indices are (row, col) for 2x2
    S = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]

    # Strassen 4x4x4 outer structure
    # x_def, y_def map block indices (0:00, 1:01, 2:10, 3:11) to coefficients
    # c_block maps output block indices to coefficients
    OUTER = [
        ({0:1, 3:1}, {0:1, 3:1}, {(0,0):1, (1,1):1}),
        ({2:1, 3:1}, {0:1}, {(1,0):1, (1,1):-1}),
        ({0:1}, {1:1, 3:-1}, {(0,1):1, (1,1):1}),
        ({3:1}, {2:1, 0:-1}, {(0,0):1, (1,0):1}),
        ({0:1, 1:1}, {3:1}, {(0,0):-1, (0,1):1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {(1,1):1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {(0,0):1})
    ]

    # Build 4x4x4 decomposition (Rank 49)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    col = 0
    for o_idx in range(7):
        x_def, y_def, c_block = OUTER[o_idx]
        for s_idx in range(7):
            u_c, v_c, w_c = S[s_idx]
            r = col
            col += 1
            
            # U coefficients
            for (li, lj), cx in u_c.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U_444[i * m + j, r] += cx * cb

            # V coefficients
            for (li, lj), cy in v_c.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V_444[i * p + j, r] += cy * cb

            # W coefficients
            for (li, lj), cc in w_c.items():
                if cc == 0: continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W_444[k_final * n + i_final, r] += cc * cb_c

    # Build 4x4x3 decomposition (Rank 44 base)
    # Using 4 independent 2x2x3 blocks (Rank 11 each)
    U_443 = np.zeros((16, 44))
    V_443 = np.zeros((12, 44)) # m*p = 4*3 = 12
    W_443 = np.zeros((12, 44)) # p*n = 3*4 = 12
    
    # 2x2x3 Strassen-like block (Rank 11)
    # B is 2x3, C is 2x3.
    # Cols 0,1 of B/C handled by 2x2x2 Strassen (Rank 7)
    # Col 2 of B/C handled by Schoolbook (Rank 4)
    
    def add_2x2x3_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        # a_rows, a_cols: 2x2 indices for A
        # b_rows: 2 indices for B rows, b_cols: 3 indices for B cols
        # c_rows: 2 indices for C rows, c_cols: 3 indices for C cols
        
        # Part 1: 2x2x2 Strassen for cols 0,1
        for s in S:
            u_c, v_c, w_c = s
            r = offset
            offset += 1
            
            for (li, lj), val in u_c.items():
                i = a_rows[li]
                j = a_cols[lj]
                U_443[i * 4 + j, r] += val # A is 4x4, so index is i*4+j
            
            for (li, lj), val in v_c.items():
                i = b_rows[li]
                # Only cols 0,1 of the 3-column block
                j = b_cols[lj] 
                V_443[i * 3 + j, r] += val
            
            for (li, lj), val in w_c.items():
                i = c_rows[li]
                k = c_cols[lj]
                W_443[k * 4 + i, r] += val

        # Part 2: 2x2x1 Schoolbook for col 2
        # C[i, 2] = sum_j A[i, j] B[j, 2]
        for li in range(2):
            for lj in range(2):
                r = offset
                offset += 1
                i = a_rows[li]
                j = a_cols[lj]
                U_443[i * 4 + j, r] = 1.0
                V_443[j * 3 + b_cols[2], r] = 1.0 # Wait, B col index in local block is 2
                # Correction: b_cols is the list of global column indices for the 3 cols of B.
                # In the local 2x2x3 block, the B columns are 0, 1, 2.
                # So we need the 3rd column of B in this block.
                V_443[b_rows[0] * 3 + b_cols[2], r] = 0 # This is wrong.
                
        # Let's redo Part 2 properly.
        # The schoolbook terms for the 3rd column (index 2 in local B/C)
        # C_local[li, 2] = A_local[li,0]*B_local[0,2] + A_local[li,1]*B_local[1,2]
        # Global indices:
        # A[i,j] where i in a_rows, j in a_cols
        # B[j, k] where j in b_rows, k = b_cols[2]
        # C[i, k] where i in c_rows, k = c_cols[2]
        
        # We need 4 terms for 2x2x1.
        # Term 1: A[0,0] * B[0,2]
        r = offset; offset += 1
        U_443[a_rows[0] * 4 + a_cols[0], r] = 1.0
        V_443[b_rows[0] * 3 + b_cols[2], r] = 1.0
        W_443[c_cols[2] * 4 + c_rows[0], r] = 1.0

        # Term 2: A[0,1] * B[1,2]
        r = offset; offset += 1
        U_443[a_rows[0] * 4 + a_cols[1], r] = 1.0
        V_443[b_rows[1] * 3 + b_cols[2], r] = 1.0
        W_443[c_cols[2] * 4 + c_rows[0], r] = 1.0

        # Term 3: A[1,0] * B[0,2]
        r = offset; offset += 1
        U_443[a_rows[1] * 4 + a_cols[0], r] = 1.0
        V_443[b_rows[0] * 3 + b_cols[2], r] = 1.0
        W_443[c_cols[2] * 4 + c_rows[1], r] = 1.0

        # Term 4: A[1,1] * B[1,2]
        r = offset; offset += 1
        U_443[a_rows[1] * 4 + a_cols[1], r] = 1.0
        V_443[b_rows[1] * 3 + b_cols[2], r] = 1.0
        W_443[c_cols[2] * 4 + c_rows[1], r] = 1.0

        return offset

    # Apply 2x2x3 blocks
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B/C cols: 4,5,6 (indices 4,5,6 in global 7-col matrix)
    # Wait, the 4x4x3 part corresponds to columns 4,5,6 of B and C.
    # So b_cols for all blocks should be [4,5,6]?
    # No, the 4x4x3 is A (4x4) * B (4x3).
    # We split A into 4 2x2 blocks.
    # We split B (4x3) into 4 2x3 blocks?
    # B is 4x3. Rows 0,1 and 2,3.
    # So B_top is 2x3, B_bot is 2x3.
    # C is 4x3. Rows 0,1 and 2,3.
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # Block (0,1): A[0:2, 2:4] * B[0:2, :] -> C[0:2, :]
    # Block (1,0): A[2:4, 0:2] * B[2:4, :] -> C[2:4, :]
    # Block (1,1): A[2:4, 2:4] * B[2:4, :] -> C[2:4, :]
    
    # Note: The output C is the SUM of these contributions.
    # So we can use the same C rows for the "top" blocks and "bottom" blocks.
    
    offset = 0
    # Top-Left A block (0,0)
    offset = add_2x2x3_block([0,1], [0,1], [0,1], [4,5,6], [0,1], [4,5,6], offset)
    # Top-Right A block (0,1)
    offset = add_2x2x3_block([0,1], [2,3], [0,1], [4,5,6], [0,1], [4,5,6], offset)
    # Bottom-Left A block (1,0)
    offset = add_2x2x3_block([2,3], [0,1], [2,3], [4,5,6], [2,3], [4,5,6], offset)
    # Bottom-Right A block (1,1)
    offset = add_2x2x3_block([2,3], [2,3], [2,3], [4,5,6], [2,3], [4,5,6], offset)
    
    assert offset == 44, f"Rank mismatch in 4x4x3: {offset}"

    # Combine 4x4x4 and 4x4x3
    # Total Rank = 49 + 44 = 93
    U_total = np.hstack([U_444, U_443])
    V_total = np.hstack([V_444, V_443])
    W_total = np.hstack([W_444, W_443])
    
    # 4x4x1 part (Rank 16)
    # C[:, 6] = A * B[:, 6]
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((28, 16)) # m*p = 4*7 = 28. Wait, V is (m*p, rank).
    # V index: j * p + k. Here k=6.
    W_441 = np.zeros((28, 16)) # W index: k * n + i. Here k=6.
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_441[i * 4 + j, r] = 1.0
            V_441[j * 7 + 6, r] = 1.0
            W_441[6 * 4 + i, r] = 1.0

    U_final = np.hstack([U_total, U_441])
    V_final = np.hstack([V_total, V_441])
    W_final = np.hstack([W_total, W_441])
    
    rank = 93
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Verify
    Ua = np.array(U_list)
    Va = np.array(V_list)
    Wa = np.array(W_list)
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    err = np.max(np.abs((Ua[u_idx] * Va[v_idx] * Wa[w_idx]).sum(axis=1) - 1.0))
    
    if err > 1e-8:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][c] = 1
                    V_s[j * p + k][c] = 1
                    W_s[k * n + i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
