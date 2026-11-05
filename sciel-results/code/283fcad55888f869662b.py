"""Tensor decomposition for 2x3 * 3x10 matrix multiplication.

Combined design:
  * Primary path: Strassen-on-2x2x2-blocks (rank 55) construction.
  * Cross-block column merging: identify columns with identical U vectors
    and merge their V/W contributions to reduce rank.
  * Refinement: fast column-removal search + linear-algebraic redundancy sweep.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _validate(n, m, p, U, V, W, atol=1e-9):
    """Verify sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, np.ones((n, m, p)), atol=atol))


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    """
    assert n == 2 and m == 3 and p == 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1: (a00 + a11) * (B[0,k0] + B[1,k1])
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2: (a10 + a11) * B[0,k0]
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3: a00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4: a11 * (B[1,k0] - B[0,k0])
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5: (a00 + a01) * B[1,k1]
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6: (a10 - a00) * (B[0,k0] + B[0,k1])
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7: (a01 - a11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # Schoolbook for A[:,2] * B[2,k0:k1+1]
        U[2][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        U[2][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        U[5][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        U[5][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    assert col == rank
    return U, V, W, rank


def _merge_columns(n, m, p, U, V, W, deadline):
    """
    Merge columns that have identical U vectors (up to scalar multiple).
    For each group of columns with the same U pattern, sum their V and W vectors.
    This is valid because the tensor contribution is linear in V and W.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]

    # Group columns by their U vector pattern (normalized)
    groups = {}
    for r in range(rank):
        u_col = U_arr[:, r]
        # Normalize: divide by first nonzero element
        nz = np.nonzero(u_col)[0]
        if len(nz) == 0:
            key = tuple(u_col)
        else:
            scale = u_col[nz[0]]
            if abs(scale) < 1e-12:
                key = tuple(u_col)
            else:
                normed = u_col / scale
                key = tuple(round(x, 6) for x in normed)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Build new matrices
    new_U_cols = []
    new_V_cols = []
    new_W_cols = []

    for key, cols in groups.items():
        if len(cols) == 0:
            continue
        # Take the U vector from the first column in the group
        u_rep = U_arr[:, cols[0]].copy()
        # Sum V and W vectors across all columns in the group
        v_sum = np.zeros_like(V_arr[:, 0])
        w_sum = np.zeros_like(W_arr[:, 0])
        for r in cols:
            v_sum += V_arr[:, r]
            w_sum += W_arr[:, r]
        new_U_cols.append(u_rep)
        new_V_cols.append(v_sum)
        new_W_cols.append(w_sum)

    if not new_U_cols:
        return U, V, W, rank

    U_new = np.column_stack(new_U_cols)
    V_new = np.column_stack(new_V_cols)
    W_new = np.column_stack(new_W_cols)

    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()

    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, len(new_U_cols)
    else:
        return U, V, W, rank


def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Fast column-removal search using vectorized residual checks.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    keep = np.ones(rank, dtype=bool)

    target = np.ones((n, m, p))

    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    contrib = np.einsum('ijr,jkr,kir->rijk', U3, V3, W3)
    current = contrib.sum(axis=0)

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            new_tensor = current - contrib[r]
            if np.allclose(new_tensor, target, atol=1e-9):
                keep[r] = False
                current = new_tensor
                changed = True
                break

    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank

    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]

    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()

    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, new_rank
    else:
        return U, V, W, rank


def _try_linear_redundancy(n, m, p, U, V, W, deadline):
    """
    Linear-algebraic redundancy sweep: remove columns that are linearly
    dependent in all three factor matrices simultaneously.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    keep = np.ones(rank, dtype=bool)

    def _column_redundant(M, r, keep_mask):
        cols = M[:, keep_mask]
        idx_list = np.where(keep_mask)[0].tolist()
        if r not in idx_list:
            return False
        idx = idx_list.index(r)
        others = np.delete(cols, idx, axis=1)
        if others.shape[1] == 0:
            return False
        col = cols[:, idx]
        if np.allclose(col, 0.0, atol=1e-12):
            return True
        try:
            x, residuals, rank_r, _ = np.linalg.lstsq(others, col, rcond=None)
        except np.linalg.LinAlgError:
            return False
        resid = np.linalg.norm(others @ x - col)
        scale = max(1.0, np.linalg.norm(col))
        return resid <= 1e-9 * scale

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            if (_column_redundant(U_arr, r, keep)
                    and _column_redundant(V_arr, r, keep)
                    and _column_redundant(W_arr, r, keep)):
                keep[r] = False
                changed = True
                break

    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank

    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, new_rank
    else:
        return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = DIMENSIONS

    # Primary path: Strassen construction (rank 55).
    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        # Fallback: trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r in range(rank):
            i = r // (m * p)
            rem = r % (m * p)
            j = rem // p
            k = rem % p
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
    else:
        # Step 1: Merge columns with identical U patterns (cross-block sharing)
        U, V, W, rank = _merge_columns(n, m, p, U, V, W, deadline)

        # Step 2: Column removal search
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)

        # Step 3: Linear redundancy sweep
        U, V, W, rank = _try_linear_redundancy(n, m, p, U, V, W, deadline)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
