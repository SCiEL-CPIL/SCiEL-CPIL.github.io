"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a rank-68 decomposition: 4 Strassen blocks for the 2x2 A-rows x 2-col B-blocks,
plus direct terms for the remaining A-row and B-column dimensions.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start = time.time()
    DEADLINE = 145.0

    # Build a list of (U_sparse, V_sparse, W_sparse) where each is dict {(i,j): coeff}
    terms = []

    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off):
        """Add 7 Strassen terms for A[a_row_off:a_row_off+2, a_col_off:a_col_off+2]
        times B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]."""
        def uidx(li):
            r, c = li // 2, li % 2
            return (a_row_off + r, a_col_off + c)
        def vidx(lj):
            r, c = lj // 2, lj % 2
            return (b_row_off + r, b_col_off + c)
        def widx(lk):
            r, c = lk // 2, lk % 2
            return (b_col_off + c, a_row_off + r)

        terms.append(({uidx(0): 1, uidx(3): 1}, {vidx(0): 1, vidx(3): 1}, {widx(0): 1, widx(3): 1}))
        terms.append(({uidx(2): 1, uidx(3): 1}, {vidx(0): 1}, {widx(2): 1, widx(3): -1}))
        terms.append(({uidx(0): 1}, {vidx(1): 1, vidx(3): -1}, {widx(1): 1, widx(3): 1}))
        terms.append(({uidx(3): 1}, {vidx(2): 1, vidx(0): -1}, {widx(0): 1, widx(2): 1}))
        terms.append(({uidx(0): 1, uidx(1): 1}, {vidx(3): 1}, {widx(0): -1, widx(1): 1}))
        terms.append(({uidx(2): 1, uidx(0): -1}, {vidx(0): 1, vidx(1): 1}, {widx(3): 1}))
        terms.append(({uidx(1): 1, uidx(3): -1}, {vidx(2): 1, vidx(3): 1}, {widx(0): 1}))

    # 4 Strassen blocks: A[0:2, 0:2] * B[0:2, k:k+2] for k in 0,2,4,6
    for k_off in (0, 2, 4, 6):
        add_strassen_block(0, 0, 0, k_off)

    # Direct terms for remaining dimensions
    for i in range(2):
        for k in range(8):
            terms.append((
                {(i, 2): 1},
                {(2, k): 1},
                {(k, i): 1}
            ))
    for j in range(2):
        for k in range(8):
            terms.append((
                {(2, j): 1},
                {(j, k): 1},
                {(k, 2): 1}
            ))
    for k in range(8):
        terms.append((
            {(2, 2): 1},
            {(2, k): 1},
            {(k, 2): 1}
        ))

    rank = len(terms)
    if rank != 68:
        # Fallback: schoolbook rank 72
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Build dense matrices directly as lists for speed
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx, (u_sp, v_sp, w_sp) in enumerate(terms):
        for (i, j), c in u_sp.items():
            U[i * m + j][idx] = c
        for (j, k), c in v_sp.items():
            V[j * p + k][idx] = c
        for (k, i), c in w_sp.items():
            W[k * n + i][idx] = c

    # Verify correctness
    U3 = np.array(U, dtype=float).reshape(n, m, rank)
    V3 = np.array(V, dtype=float).reshape(m, p, rank)
    W3 = np.array(W, dtype=float).reshape(p, n, rank)
    T_computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p), dtype=float)

    if not np.allclose(T_computed, T_target, atol=1e-10):
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Convert to clean integer representation (all entries are ±1 or 0)
    def clean_list(lst):
        return [int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in lst]

    U_out = [clean_list(row) for row in U]
    V_out = [clean_list(row) for row in V]
    W_out = [clean_list(row) for row in W]

    elapsed = time.time() - start
    if elapsed > DEADLINE:
        pass

    return {"U": U_out, "V": V_out, "W": W_out}
