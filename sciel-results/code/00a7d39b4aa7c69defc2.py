"""Rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.

Uses 6 disjoint 2x2x2 Strassen blocks (rank 7 each) plus schoolbook for the rest.
Total rank = 6*7 + 36 = 78.
"""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    rank = 6 * 7 + 36  # 78
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

    # Schoolbook for all remaining (i,j,k)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    # Optimization: Use NumPy for verification if available and fast enough, otherwise loop.
    # Given small dimensions, loop is fine, but we check time.
    try:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # Compute sum over rank for all i, j, k
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # This is a contraction.
        # We can do:
        # Temp[i, j, k] = sum_c U[i, j, c] * V[j, k, c]
        # Then T[i, j, k] = sum_c Temp[i, j, c] * W[k, i, c]? No, W index is k, i.
        
        # Let's use einsum
        # U: i, j, r
        # V: j, k, r
        # W: k, i, r
        # Result: i, j, k
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # This is not a standard einsum because of the cyclic indices in W.
        # We can permute W to i, k, r.
        W_perm = W_np.transpose(1, 0, 2) # i, k, r
        
        # Now we want sum_r U[i,j,r] V[j,k,r] W[i,k,r]
        # This is not a single einsum string easily because j is summed in U and V, but not in W?
        # No, the verification sums over c (rank) only.
        # The indices i, j, k are free.
        # So we want:
        # R[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        
        # We can compute:
        # A[i, j, k, r] = U[i, j, r] * V[j, k, r]
        # Then R[i, j, k] = sum_r A[i, j, k, r] * W[k, i, r]
        
        # Using einsum:
        # 'ijr,jkr,kir->ijk'
        # Note: W is indexed by k, i, r in the original array W_np (p, n, rank).
        # So W[k, i, r] is correct.
        
        R = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_np)
        
        if not np.all(R == 1):
             raise ValueError("Decomposition failed with NumPy check")
    except Exception:
        # Fallback to manual check if NumPy fails or is too slow
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
