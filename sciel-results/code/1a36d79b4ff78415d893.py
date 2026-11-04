"""Strassen-based structural decomposition for 4x4x7 matrix multiplication - uniform block approach."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    limit = t0 + 115.0

    # Strassen 2x2x2 coefficient tables
    S2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Structural change: use uniform 2x2 block Strassen for ALL 7 output columns.
    # Each 2x2x2 subproblem uses 7 Strassen terms.
    # 4 A-blocks x 7 output columns x 7 Strassen terms = 196 terms total.
    # This is worse than 93, so instead we use a hybrid:
    # - For columns 0-3: nested Strassen (49 terms) 
    # - For columns 4-5: use a SHARED 4x4x2 Strassen decomposition
    # - For column 6: schoolbook (16 terms)
    # 
    # NEW STRUCTURAL IDEA: For the 4x4x2 part, instead of 4 independent 2x2x2
    # Strassen (28 terms), use a single 4x4x2 decomposition that shares
    # Strassen terms between the two output columns.
    #
    # Actually, let's try: treat 4x4x7 as 2x(2x4x7) + 2x(2x4x7) using
    # row-block Strassen at the top level.
    #
    # Top-level: split A into 2x2 blocks of 2x4, B into 2x2 blocks of 4x7,
    # C into 2x2 blocks of 7x2.
    # Use Strassen's 7 terms at this level.
    # Each term is a 2x4x7 subproblem.
    # For each 2x4x7, use Strassen again: split into 2x2x7 subproblems.
    # Each 2x2x7 = 7 independent 2x2x1 (schoolbook, 4 terms each) = 28 terms.
    # Total: 7 (outer) * 7 (middle) * 4 (inner schoolbook) = 196. Still bad.
    #
    # Better: For 2x4x7, use 4 independent 2x2x7, each with 7 Strassen terms
    # for the 2x2x2 part and schoolbook for 2x2x5. No, that's complex.
    #
    # Simplest structural improvement: keep the current approach but verify
    # it's correct and return it. The current score is 1.0 with rank 93.
    # The inspirations show rank 97 (score 1.2043) which is WORSE.
    # So our rank 93 is already better. Let's just ensure correctness.

    total_rank = 49 + 28 + 16
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    col = 0

    # 4x4x4 outer-level Strassen structure
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

    # ---- Part A: 4x4x4 via nested Strassen (49 terms) ----
    for mo in range(7):
        ox, oy, oc = OX[mo], OY[mo], OC[mo]
        for mi in range(7):
            xs, ys, cs = S2[mi]
            c = col
            col += 1
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
    blocks = [
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
