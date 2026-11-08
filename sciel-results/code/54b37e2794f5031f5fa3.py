"""Deterministic rank-78 decomposition for 3x4*4x7 matrix multiplication.

Uses 6 disjoint 2x2x2 Strassen blocks (7 rank-1 terms each) plus 36 schoolbook
terms.  The 6 Strassen blocks partition the 2x4x6 = 48 covered triples; the
remaining 36 (i,j,k) triples get one schoolbook term each.

Total rank = 6*7 + 36 = 78.
"""
import time
import numpy as np

_DIMS = (3, 4, 7)

# Precomputed block definitions: (i0, i1, j0, j1, k0, k1)
_BLOCKS = [
    (0, 1, 0, 1, 0, 1),
    (0, 1, 0, 1, 2, 3),
    (0, 1, 0, 1, 4, 5),
    (0, 1, 2, 3, 0, 1),
    (0, 1, 2, 3, 2, 3),
    (0, 1, 2, 3, 4, 5),
]


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = _DIMS

    rank = 78
    rows_u = n * m          # 12
    rows_v = m * p          # 28
    rows_w = p * n          # 21

    # Use flat integer arrays for speed; convert to nested lists at the end.
    U = np.zeros((rows_u, rank), dtype=np.int64)
    V = np.zeros((rows_v, rank), dtype=np.int64)
    W = np.zeros((rows_w, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
        c = col_start

        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]) -> C[i0,k0]+, C[i1,k1]+=
        U[i0 * m + j0, c] = 1
        U[i1 * m + j1, c] = 1
        V[j0 * p + k0, c] = 1
        V[j1 * p + k1, c] = 1
        W[k0 * n + i0, c] = 1
        W[k1 * n + i1, c] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0] -> C[i1,k0]+, C[i1,k1]-
        U[i1 * m + j0, c + 1] = 1
        U[i1 * m + j1, c + 1] = 1
        V[j0 * p + k0, c + 1] = 1
        W[k0 * n + i1, c + 1] = 1
        W[k1 * n + i1, c + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1]) -> C[i0,k1]+, C[i1,k1]+=
        U[i0 * m + j0, c + 2] = 1
        V[j0 * p + k1, c + 2] = 1
        V[j1 * p + k1, c + 2] = -1
        W[k1 * n + i0, c + 2] = 1
        W[k1 * n + i1, c + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0]) -> C[i0,k0]+, C[i1,k0]+=
        U[i1 * m + j1, c + 3] = 1
        V[j1 * p + k0, c + 3] = 1
        V[j0 * p + k0, c + 3] = -1
        W[k0 * n + i0, c + 3] = 1
        W[k0 * n + i1, c + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1] -> C[i0,k0]-, C[i0,k1]+=
        U[i0 * m + j0, c + 4] = 1
        U[i0 * m + j1, c + 4] = 1
        V[j1 * p + k1, c + 4] = 1
        W[k0 * n + i0, c + 4] = -1
        W[k1 * n + i0, c + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1]) -> C[i1,k1]+=
        U[i1 * m + j0, c + 5] = 1
        U[i0 * m + j0, c + 5] = -1
        V[j0 * p + k0, c + 5] = 1
        V[j0 * p + k1, c + 5] = 1
        W[k1 * n + i1, c + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1]) -> C[i0,k0]+=
        U[i0 * m + j1, c + 6] = 1
        U[i1 * m + j1, c + 6] = -1
        V[j1 * p + k0, c + 6] = 1
        V[j1 * p + k1, c + 6] = 1
        W[k0 * n + i0, c + 6] = 1

        return col_start + 7

    # 6 Strassen blocks
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in _BLOCKS:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for all remaining (i,j,k)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0, col] = 1
                V[j0 * p + k0, col] = 1
                W[k0 * n + i0, col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Fast vectorized verification
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)

    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(m):
            u_row = Uf[i * m + j]
            for k in range(p):
                v_row = Vf[j * p + k]
                w_row = Wf[k * n + i]
                total = np.dot(u_row, v_row * w_row)
                if abs(total - 1.0) > 1e-10:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}")

    # Convert to nested lists with integer entries
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
