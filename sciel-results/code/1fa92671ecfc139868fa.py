"""Strassen-based structural decomposition for 4x4x7 matrix multiplication - alternate construction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    limit = t0 + 115.0

    # Precompute Strassen 2x2x2 coefficient tables as flat arrays
    # Each term: (x_flat, y_flat, c_flat) where flat is 4-element list for 2x2
    S2 = [
        # term 0
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # term 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # term 2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # term 3
        ([0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]),
        # term 4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # term 5
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # term 6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # 4x4x4 outer-level Strassen structure
    # outer_x[m]: dict mapping block_idx (0-3) -> coeff
    # outer_y[m]: dict mapping block_idx (0-3) -> coeff
    # outer_c[m]: dict mapping (row_block, col_block) -> coeff
    OX = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    OY = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    OC = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    total_rank = 49 + 28 + 16
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    col = 0

    # ---- Part A: 4x4x4 via nested Strassen (49 terms) ----
    for mo in range(7):
        ox, oy, oc = OX[mo], OY[mo], OC[mo]
        for mi in range(7):
            xs, ys, cs = S2[mi]
            c = col
            col += 1
            # U: A entries
            for bi in range(2):
                for bj in range(2):
                    bidx = 2 * bi + bj
                    cb = ox.get(bidx, 0)
                    if cb == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            sx = xs[li * 2 + lj]
                            if sx == 0:
                                continue
                            i = 2 * bi + li
                            j = 2 * bj + lj
                            U[i * m + j, c] += sx * cb
            # V: B entries (first 4 columns)
            for bi in range(2):
                for bj in range(2):
                    bidx = 2 * bi + bj
                    cb = oy.get(bidx, 0)
                    if cb == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            sy = ys[li * 2 + lj]
                            if sy == 0:
                                continue
                            i = 2 * bi + li
                            j = 2 * bj + lj
                            V[i * p + j, c] += sy * cb
            # W: C entries (first 4 rows, first 4 cols)
            for (br, bc), cc_outer in oc.items():
                for li in range(2):
                    for lj in range(2):
                        sc = cs[li * 2 + lj]
                        if sc == 0:
                            continue
                        i_f = 2 * br + li
                        k_f = 2 * bc + lj
                        W[k_f * n + i_f, c] += sc * cc_outer

    # ---- Part B: 4x4x2 via four independent 2x2x2 Strassen (28 terms) ----
    # B columns 4,5; C rows 0-1 get top half, rows 2-3 get bottom half
    blocks = [
        # (A_rows, A_cols, B_rows, B_cols, C_rows, C_cols)
        ([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [4, 5], [0, 1], [4, 5]),
        ([2, 3], [0, 1], [0, 1], [4, 5], [2, 3], [4, 5]),
        ([2, 3], [2, 3], [2, 3], [4, 5], [2, 3], [4, 5]),
    ]
    for (ar, ac, br, bc, cr, cc) in blocks:
        for mi in range(7):
            xs, ys, cs = S2[mi]
            c = col
            col += 1
            for li in range(2):
                for lj in range(2):
                    sx = xs[li * 2 + lj]
                    if sx:
                        U[ar[li] * m + ac[lj], c] += sx
                    sy = ys[li * 2 + lj]
                    if sy:
                        V[br[li] * p + bc[lj], c] += sy
                    sc = cs[li * 2 + lj]
                    if sc:
                        W[cc[lj] * n + cr[li], c] += sc

    # ---- Part C: 4x4x1 schoolbook for column 6 (16 terms) ----
    for i in range(4):
        for j in range(4):
            c = col
            col += 1
            U[i * m + j, c] = 1.0
            V[j * p + 6, c] = 1.0
            W[6 * n + i, c] = 1.0

    assert col == total_rank, f"rank mismatch: {col} != {total_rank}"

    # Quick self-check
    Ua, Va, Wa = U, V, W
    ii = np.repeat(np.arange(n), m * p)
    jj = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n * m)
    ui = ii * m + jj
    vi = jj * p + kk
    wi = kk * n + ii
    check = (Ua[ui, :] * Va[vi, :] * Wa[wi, :]).sum(axis=1)
    if np.max(np.abs(check - 1.0)) > 1e-9:
        # Fallback to schoolbook
        rs = n * m * p
        Us = np.zeros((n * m, rs))
        Vs = np.zeros((m * p, rs))
        Ws = np.zeros((p * n, rs))
        cc2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, cc2] = 1
                    Vs[j * p + k, cc2] = 1
                    Ws[k * n + i, cc2] = 1
                    cc2 += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    if time.time() > limit:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
