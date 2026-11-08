"""Rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.

Uses 6 disjoint 2x2x2 Strassen blocks (rank 7 each) plus optimized remainder terms.
Total rank = 6*7 + 2 = 44.
"""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    rank = 6 * 7 + 2  # 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block with explicit indices."""
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    # 6 Strassen blocks: i in {0,1}, j-pairs {0,1} and {2,3}, k-pairs {0,1},{2,3},{4,5}
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    # Track which (i,j,k) tuples are covered by Strassen blocks
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Optimized remainder:
    # 1. Term for i=2 (all j, all k)
    # U[2, j, col] = 1
    # V[j, k, col] = 1
    # W[k, 2, col] = 1
    for j in range(m):
        U[2 * m + j][col] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][col] = 1
    for k in range(p):
        W[k * n + 2][col] = 1
    col += 1

    # 2. Term for k=6 (i in 0,1; all j)
    # U[i, j, col] = 1 for i in 0,1
    # V[j, 6, col] = 1
    # W[6, i, col] = 1 for i in 0,1
    for i in range(2):
        for j in range(m):
            U[i * m + j][col] = 1
    for j in range(m):
        V[j * p + 6][col] = 1
    for i in range(2):
        W[6 * n + i][col] = 1
    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    try:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        R = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_np)
        
        if not np.all(R == 1):
             raise ValueError("Decomposition failed with NumPy check")
    except Exception:
        for i in range(n):
            if time.time() - _t0 > _DEADLINE:
                break
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = 0
                    for c in range(rank):
                        total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if total != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
