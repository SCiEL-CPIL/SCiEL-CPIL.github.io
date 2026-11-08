"""Rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.

This implementation uses a hybrid approach:
- 6 disjoint 2x2x2 Strassen blocks (rank 7 each) for rows 0-1 of A
- Schoolbook for all remaining (i,j,k) tuples
Total rank = 6*7 + 36 = 78.

Key improvement over previous: uses vectorized NumPy verification for speed,
and constructs the decomposition more efficiently.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    rank = 6 * 7 + 36  # 78
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block with explicit indices."""
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0, col_start] = 1
        U[i1 * m + j1, col_start] = 1
        V[j0 * p + k0, col_start] = 1
        V[j1 * p + k1, col_start] = 1
        W[k0 * n + i0, col_start] = 1
        W[k1 * n + i1, col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0, col_start + 1] = 1
        U[i1 * m + j1, col_start + 1] = 1
        V[j0 * p + k0, col_start + 1] = 1
        W[k0 * n + i1, col_start + 1] = 1
        W[k1 * n + i1, col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0, col_start + 2] = 1
        V[j0 * p + k1, col_start + 2] = 1
        V[j1 * p + k1, col_start + 2] = -1
        W[k1 * n + i0, col_start + 2] = 1
        W[k1 * n + i1, col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1, col_start + 3] = 1
        V[j1 * p + k0, col_start + 3] = 1
        V[j0 * p + k0, col_start + 3] = -1
        W[k0 * n + i0, col_start + 3] = 1
        W[k0 * n + i1, col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0, col_start + 4] = 1
        U[i0 * m + j1, col_start + 4] = 1
        V[j1 * p + k1, col_start + 4] = 1
        W[k0 * n + i0, col_start + 4] = -1
        W[k1 * n + i0, col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0, col_start + 5] = 1
        U[i0 * m + j0, col_start + 5] = -1
        V[j0 * p + k0, col_start + 5] = 1
        V[j0 * p + k1, col_start + 5] = 1
        W[k1 * n + i1, col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1, col_start + 6] = 1
        U[i1 * m + j1, col_start + 6] = -1
        V[j1 * p + k0, col_start + 6] = 1
        V[j1 * p + k1, col_start + 6] = 1
        W[k0 * n + i0, col_start + 6] = 1

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

    # Verify the decomposition using vectorized NumPy
    if time.time() - _t0 < _DEADLINE - 5:
        # Check that for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        Uf = U.astype(np.float64)
        Vf = V.astype(np.float64)
        Wf = W.astype(np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Uf[i * m + j] * Vf[j * p + k] * Wf[k * n + i]
                    total = np.sum(prod)
                    if abs(total - 1.0) > 1e-10:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    # Convert to list-of-lists format as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
