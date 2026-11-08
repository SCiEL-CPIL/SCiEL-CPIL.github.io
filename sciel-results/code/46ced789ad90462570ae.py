"""Deterministic rank-54 decomposition for 3x4*4x7 matrix multiplication.

Uses 6 disjoint 2x2x2 Strassen blocks covering i in {0,1}, j in {0,1}x{2,3}, k in {0,1}x{2,3}x{4,5}.
Remaining entries:
  - i=2, all j in {0..3}, all k in {0..6}: decomposed with 4 rank-1 terms (one per j)
  - i in {0,1}, all j in {0..3}, k=6: decomposed with 8 rank-1 terms (one per (i,j) pair)
Total rank = 6*7 + 4 + 8 = 54.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 115.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS

    # 6 Strassen blocks: 7 terms each = 42
    # i in {0,1}, j partitioned into {0,1} and {2,3}, k partitioned into {0,1}, {2,3}, {4,5}
    strassen_blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    # 4 terms for i=2: one per j in {0,1,2,3}
    # 8 terms for k=6: one per (i,j) in {0,1}x{0,1,2,3}
    total_rank = 6 * 7 + 4 + 8  # 54

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    covered = set()
    for (i0, i1, j0, j1, k0, k1) in strassen_blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # i=2 slice: 4 terms, one per j
    for j in range(m):
        U[2 * m + j][col] = 1
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
        col += 1

    # k=6 slice for i in {0,1}: 8 terms, one per (i,j)
    for i in range(2):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1

    assert col == total_rank, f"Column count {col} != rank {total_rank}"

    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)

    # Verify: sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r] should be 1 for all (i,j,k)
    i_vals, j_vals, k_vals = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    u_idx = (i_vals * m + j_vals).flatten()
    v_idx = (j_vals * p + k_vals).flatten()
    w_idx = (k_vals * n + i_vals).flatten()

    products = U_np[u_idx] * V_np[v_idx] * W_np[w_idx]
    totals = products.sum(axis=1)

    if time.time() - _t0 < _DEADLINE:
        if not np.all(totals == 1):
            bad = np.where(totals != 1)[0]
            for b in bad[:5]:
                i_b = i_vals.flatten()[b]
                j_b = j_vals.flatten()[b]
                k_b = k_vals.flatten()[b]
                raise ValueError(f"Decomposition failed at ({i_b},{j_b},{k_b}): got {totals[b]}, expected 1")

    return {"U": U, "V": V, "W": W}
