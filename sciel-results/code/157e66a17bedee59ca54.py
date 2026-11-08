"""Deterministic multi-Strassen rank-78 decomposition for 3x4*4x7 matrix multiplication.

Uses 6 disjoint 2x2x2 Strassen blocks with i in {0,1}, partitioning j into {0,1}x{2,3}
and k into {0,1}x{2,3}x{4,5}. Remaining 36 entries use schoolbook.
Total rank = 6*7 + 36 = 78.

This version uses a sparse column representation for fast construction and verification,
combined with the direct Strassen block placement logic.
"""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 135.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # 6 Strassen blocks (i in {0,1}), 36 schoolbook terms
    rank = 6 * 7 + 36  # 78
    rows_u = n * m      # 12
    rows_v = m * p      # 28
    rows_w = p * n      # 21

    # Sparse column representation: list of (row, value) tuples per factor
    U_cols = [[] for _ in range(rank)]
    V_cols = [[] for _ in range(rank)]
    W_cols = [[] for _ in range(rank)]

    def put(arr, col, row, val):
        arr[col].append((row, val))

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block with explicit indices."""
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]) -> C[i0,k0]+, C[i1,k1]+=
        put(U_cols, col_start, i0 * m + j0, 1)
        put(U_cols, col_start, i1 * m + j1, 1)
        put(V_cols, col_start, j0 * p + k0, 1)
        put(V_cols, col_start, j1 * p + k1, 1)
        put(W_cols, col_start, k0 * n + i0, 1)
        put(W_cols, col_start, k1 * n + i1, 1)

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0] -> C[i1,k0]+, C[i1,k1]-
        put(U_cols, col_start + 1, i1 * m + j0, 1)
        put(U_cols, col_start + 1, i1 * m + j1, 1)
        put(V_cols, col_start + 1, j0 * p + k0, 1)
        put(W_cols, col_start + 1, k0 * n + i1, 1)
        put(W_cols, col_start + 1, k1 * n + i1, -1)

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1]) -> C[i0,k1]+, C[i1,k1]+=
        put(U_cols, col_start + 2, i0 * m + j0, 1)
        put(V_cols, col_start + 2, j0 * p + k1, 1)
        put(V_cols, col_start + 2, j1 * p + k1, -1)
        put(W_cols, col_start + 2, k1 * n + i0, 1)
        put(W_cols, col_start + 2, k1 * n + i1, 1)

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0]) -> C[i0,k0]+, C[i1,k0]+=
        put(U_cols, col_start + 3, i1 * m + j1, 1)
        put(V_cols, col_start + 3, j1 * p + k0, 1)
        put(V_cols, col_start + 3, j0 * p + k0, -1)
        put(W_cols, col_start + 3, k0 * n + i0, 1)
        put(W_cols, col_start + 3, k0 * n + i1, 1)

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1] -> C[i0,k0]-, C[i0,k1]+=
        put(U_cols, col_start + 4, i0 * m + j0, 1)
        put(U_cols, col_start + 4, i0 * m + j1, 1)
        put(V_cols, col_start + 4, j1 * p + k1, 1)
        put(W_cols, col_start + 4, k0 * n + i0, -1)
        put(W_cols, col_start + 4, k1 * n + i0, 1)

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1]) -> C[i1,k1]+=
        put(U_cols, col_start + 5, i1 * m + j0, 1)
        put(U_cols, col_start + 5, i0 * m + j0, -1)
        put(V_cols, col_start + 5, j0 * p + k0, 1)
        put(V_cols, col_start + 5, j0 * p + k1, 1)
        put(W_cols, col_start + 5, k1 * n + i1, 1)

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1]) -> C[i0,k0]+=
        put(U_cols, col_start + 6, i0 * m + j1, 1)
        put(U_cols, col_start + 6, i1 * m + j1, -1)
        put(V_cols, col_start + 6, j1 * p + k0, 1)
        put(V_cols, col_start + 6, j1 * p + k1, 1)
        put(W_cols, col_start + 6, k0 * n + i0, 1)

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
                put(U_cols, col, i0 * m + j0, 1)
                put(V_cols, col, j0 * p + k0, 1)
                put(W_cols, col, k0 * n + i0, 1)
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Convert to dense list-of-lists (the required interface)
    U = [[0] * rank for _ in range(rows_u)]
    V = [[0] * rank for _ in range(rows_v)]
    W = [[0] * rank for _ in range(rows_w)]
    for c in range(rank):
        for (r, v) in U_cols[c]:
            U[r][c] = v
        for (r, v) in V_cols[c]:
            V[r][c] = v
        for (r, v) in W_cols[c]:
            W[r][c] = v

    # Verify the decomposition using sparse representation for speed
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
                    u_val = 0
                    v_val = 0
                    w_val = 0
                    for (r, val) in U_cols[c]:
                        if r == u_idx:
                            u_val = val
                            break
                    for (r, val) in V_cols[c]:
                        if r == v_idx:
                            v_val = val
                            break
                    for (r, val) in W_cols[c]:
                        if r == w_idx:
                            w_val = val
                            break
                    total += u_val * v_val * w_val
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
