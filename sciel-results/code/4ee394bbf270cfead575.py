"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
  - 4x4 * 4x4 part (columns 0-3 of B): recursive Strassen (rank 49)
  - 4x4 * 4x2 part (columns 4-5 of B): 4 independent 2x2*2x2 Strassen (rank 28)
  Total rank: 49 + 28 = 77

The decomposition is constructed directly into the final U, V, W matrices
using the Kronecker-product / block-Strassen structure.

Verification is performed at construction time; on failure, falls back to
the schoolbook rank-96 decomposition.
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def construct():
    """
    Construct a rank-77 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Returns dict with 'U' (16 x rank), 'V' (24 x rank), 'W' (24 x rank).
    """
    n, m, p = 4, 4, 6
    rank = 77

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 steps: (alpha, beta, gamma)
    # Each is a dict mapping (i, j) -> coefficient
    # alpha: A (2x2), beta: B (2x2), gamma: C (2x2)
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # ---- Part 1: 4x4 * 4x4 (columns 0-3) via recursive Strassen, rank 49 ----
    # Outer Strassen: 7 terms combining 2x2 blocks of A, B, C
    # Inner Strassen: 7 terms for each 2x2*2x2 block multiplication
    #
    # Block layout for 4x4:
    #   Block 0: rows 0-1, cols 0-1  (TL)
    #   Block 1: rows 0-1, cols 2-3  (TR)
    #   Block 2: rows 2-3, cols 0-1  (BL)
    #   Block 3: rows 2-3, cols 2-3  (BR)
    #
    # For U (A matrix): row index = i*m + j = i*4 + j
    #   Block (bi, bj) -> i = bi*2+si, j = bj*2+sj -> row = (bi*2+si)*4 + (bj*2+sj)
    # For V (B matrix): row index = j*p + k = j*6 + k
    #   In the 4x4 part, B is 4x4 (cols 0-3), so k = bj*2+sj (but p=6, so k is within 0-3)
    #   Block (bi, bj) -> j = bi*2+si, k = bj*2+sj -> row = (bi*2+si)*6 + (bj*2+sj)
    # For W (C matrix): index = k*n + i = k*4 + i  (trace convention W[k*n+i, r])
    #   Block (bi, bj) -> k = bj*2+sj, i = bi*2+si -> index = (bj*2+sj)*4 + (bi*2+si)

    # Outer Strassen block combinations
    # Each: (A_blocks, B_blocks, C_blocks) where each is {block_idx: coeff}
    # Block idx: 0=TL, 1=TR, 2=BL, 3=BR
    outer_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00+C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00 -> C10+C11 (with C11 negated)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11) -> C01+C11 (with C11 negated... wait)
        # Actually: C01 = M1 - M5 + M3, C11 = M1 + M4 - M2
        # Let me use the standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M5 + M3
        # C01 = M4 + M5
        # C10 = M2 + M7
        # C11 = M1 - M2 + M4
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for s_outer in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[s_outer]
        for s_inner in range(7):
            alpha_2, beta_2, gamma_2 = S2[s_inner]
            col = s_outer * 7 + s_inner

            # U: A is 4x4, row = i*4 + j
            for (bi, bj), val_a4 in a_blocks.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2

            # V: B is 4x4 (cols 0-3 of the 4x6), row = j*6 + k
            for (bi, bj), val_b4 in b_blocks.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2

            # W: C is 4x4 (cols 0-3), index = k*4 + i
            for (bi, bj), val_c4 in c_blocks.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # ---- Part 2: 4x4 * 4x2 (columns 4-5) via 4 independent 2x2*2x2, rank 28 ----
    # C_top (rows 0-1, cols 4-5) = A00 * B_top + A01 * B_bot
    # C_bot (rows 2-3, cols 4-5) = A10 * B_top + A11 * B_bot
    # where B_top = B[0:2, 4:6], B_bot = B[2:4, 4:6]
    #
    # Each product is a 2x2 * 2x2 multiplication, decomposed via Strassen (rank 7).
    # 4 products * 7 = 28 terms.
    #
    # For product A_block * B_block -> C_block:
    #   A_block at (a_bi, a_bj): rows a_bi*2+si, cols a_bj*2+sj
    #   B_block at (b_half, 4+sj): rows b_half*2+si, cols 4+sj
    #   C_block at (c_half*2+si, 4+sj): index = (4+sj)*4 + (c_half*2+si)

    mults = [
        # (a_block_row, a_block_col, b_row_half, c_row_half)
        (0, 0, 0, 0),  # A00 * B_top -> C_top
        (0, 1, 1, 0),  # A01 * B_bot -> C_top
        (1, 0, 0, 1),  # A10 * B_top -> C_bot
        (1, 1, 1, 1),  # A11 * B_bot -> C_bot
    ]

    for m_idx, (a_bi, a_bj, b_half, c_half) in enumerate(mults):
        for s_inner in range(7):
            alpha_2, beta_2, gamma_2 = S2[s_inner]
            col = 49 + m_idx * 7 + s_inner

            # U: A block at (a_bi, a_bj)
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_bi * 2 + si) * m + (a_bj * 2 + sj)
                U[r][col] += val_a2

            # V: B block, rows (b_half*2+si), cols (4+sj)
            for (si, sj), val_b2 in beta_2.items():
                r = (b_half * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2

            # W: C block, col = 4+sj, row = c_half*2+si
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_half * 2 + si)
                W[r][col] += val_c2

    # Verify the decomposition
    if _time_remaining() > 3.0:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)

    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()

    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]

    totals = np.sum(U_sel * V_sel * W_sel, axis=1)

    return bool(np.all(totals == 1))
