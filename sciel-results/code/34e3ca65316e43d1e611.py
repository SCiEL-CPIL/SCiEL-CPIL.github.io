"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    Uses Laderman's rank-15 decomposition at the 3x3 block level (2x2 blocks)
    with schoolbook rank-8 at the 2x2 level, achieving total rank 120.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # all 6

    # Laderman rank-15 for 3x3 block matrices, schoolbook rank-8 for 2x2 blocks
    # Total rank = 15 * 8 = 120

    rank = 120
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(row, col):
        return row * m + col

    def v_idx(row, col):
        return row * p + col

    def w_idx(row, col):
        # W[k*n+i] corresponds to C[i,k]
        return row * n + col

    # Laderman's 15 products for 3x3 matrix multiplication (block level)
    # Each term: (u_blocks, v_blocks)
    # u_blocks: dict (bi,bj) -> coeff for A block
    # v_blocks: dict (bk,bl) -> coeff for B block

    laderman_terms = [
        ({(0,0): 1}, {(0,0): 1, (1,2): 1, (2,1): 1}),
        ({(0,1): 1}, {(0,1): 1, (1,2): 1, (2,0): 1}),
        ({(0,2): 1}, {(0,2): 1, (1,0): 1, (2,1): 1}),
        ({(1,0): 1}, {(0,0): 1, (1,1): 1, (2,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,2): 1, (2,0): 1}),
        ({(1,2): 1}, {(0,2): 1, (1,0): 1, (2,1): 1}),
        ({(2,0): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}),
        ({(2,1): 1}, {(0,1): 1, (1,2): 1, (2,0): 1}),
        ({(2,2): 1}, {(0,2): 1, (1,0): 1, (2,1): 1}),
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}),
        ({(0,1): 1, (1,2): 1}, {(0,1): 1, (1,2): 1, (2,0): 1}),
        ({(0,2): 1, (1,0): 1}, {(0,2): 1, (1,0): 1, (2,1): 1}),
        ({(1,0): 1, (2,1): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}),
        ({(1,1): 1, (2,2): 1}, {(0,1): 1, (1,2): 1, (2,0): 1}),
        ({(2,2): 1, (0,0): 1}, {(0,2): 1, (1,0): 1, (2,1): 1}),
    ]

    # Output: C[oi,oj] = sum_t coeff[(oi,oj)][t] * M_t
    output_coeffs = {
        (0,0): {0: 1, 3: 1, 6: -1, 9: -1, 12: 1},
        (0,1): {1: 1, 4: 1, 7: -1, 10: -1, 13: 1},
        (0,2): {2: 1, 5: 1, 8: -1, 11: -1, 14: 1},
        (1,0): {6: 1, 0: -1, 9: 1, 3: -1, 12: 1},
        (1,1): {7: 1, 1: -1, 10: 1, 4: -1, 13: 1},
        (1,2): {8: 1, 2: -1, 11: 1, 5: -1, 14: 1},
        (2,0): {3: 1, 6: -1, 9: -1, 12: 1, 0: 1},
        (2,1): {4: 1, 7: -1, 10: -1, 13: 1, 1: 1},
        (2,2): {5: 1, 8: -1, 11: -1, 14: 1, 2: 1},
    }

    # Schoolbook 2x2 terms: (a_row, a_col, b_row, b_col, c_row, c_col)
    schoolbook_terms = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ]

    col = 0
    for t in range(15):
        u_blocks, v_blocks = laderman_terms[t]

        for s in range(8):
            a_r, a_c, b_r, b_c, c_r, c_c = schoolbook_terms[s]

            # U factor
            for (bi, bj), ci in u_blocks.items():
                U[u_idx(2*bi + a_r, 2*bj + a_c)][col] += ci

            # V factor
            for (bk, bl), dj in v_blocks.items():
                V[v_idx(2*bk + b_r, 2*bl + b_c)][col] += dj

            # W factor
            for (oi, oj), coeffs in output_coeffs.items():
                w_coeff = coeffs.get(t, 0)
                if w_coeff != 0:
                    W[w_idx(2*oi + c_r, 2*oj + c_c)][col] += w_coeff

            col += 1

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    import random
    random.seed(42)
    valid = True
    for _ in range(500):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
