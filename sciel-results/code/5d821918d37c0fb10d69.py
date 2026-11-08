import numpy as np
import time

DIMENSIONS = [2, 4, 6]
DEADLINE = 145.0  # safety margin before 150s kill


def _build_rank42():
    """Build the known rank-42 block-Strassen solution (matmul tensor, all-ones)."""
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_s = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_s = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    col = 0
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_s[s].items():
                    u_idx = i_loc * m + (a_block * 2 + j_loc)
                    U[u_idx, col] += coeff
                for (j_loc, k_loc), coeff in V_s[s].items():
                    v_idx = (a_block * 2 + j_loc) * p + (b_block * 2 + k_loc)
                    V[v_idx, col] += coeff
                for (i_loc, k_loc), coeff in W_s[s].items():
                    w_idx = (b_block * 2 + k_loc) * n + i_loc
                    W[w_idx, col] += coeff
                col += 1
    return U, V, W


def _tensor_error(U, V, W, tol=1e-8):
    """Return Frobenius error of the decomposition vs the all-ones matmul tensor.
    The matmul tensor T[i,j,k]=1 (since (AB)[i,k]=sum_j A[i,j]B[j,k])."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Vectorized: for each r, outer of U_r (n*m), V_r (m*p), W_r (p*n)
    # Reshape U->(n,m,r), V->(m,p,r), W->(p,n,r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
    # = einsum('ijr,jkr,kir->ijk')
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return float(np.linalg.norm(T - 1.0))


def _tensor_check(U, V, W, tol=1e-8):
    return _tensor_error(U, V, W) < tol


def _round_to_halfint(M, tol=1e-6):
    doubled = 2.0 * M
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        return rounded / 2.0
    return None


def _als_polish(U, V, W, max_iters=60, tol=1e-10, time_limit=None):
    """Alternating least squares to reduce the all-ones tensor error.
    For fixed V,W: U[i,j,r] should satisfy sum_r U[i,j,r]*(V[j,k,r]*W[k,i,r]) = 1.
    This is a linear least-squares problem per (i,j) over the rank dimension."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    start = time.time()
    for it in range(max_iters):
        if time_limit is not None and time.time() - start > time_limit:
            break
        # Update U: for each (i,j), solve min_U sum_k (1 - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r])^2
        # = min over U[i,j,:] of ||1 - A[i,j] @ U[i,j,:]|| where A[i,j,r,k]=V[j,k,r]*W[k,i,r]
        # Build A[i,j,r,k] = V3[j,k,r]*W3[k,i,r] -> shape (n,m,r,p)
        # A[i,j,r,k] = V3[j,k,r] * W3[k,i,r]
        A = np.einsum('jkr,kir->ijrk', V3, W3, optimize=True)  # (n,m,r,p)
        # For each (i,j): residual_k = 1 - sum_r A[i,j,r,k]*U[i,j,r]; minimize ||residual||^2
        # Linear system: (A^T A)[i,j] @ U[i,j,:] = A^T @ 1
        ATA = np.einsum('ijrk,ijrs->ijks', A, A, optimize=True)  # (n,m,r,s)
        AT1 = np.einsum('ijrk->ijr', A)  # (n,m,r)  [sum over k of A[i,j,r,k]]
        # Solve ATA[i,j] @ U[i,j,:] = AT1[i,j]
        U_new = np.empty_like(U3)
        for i in range(n):
            for j in range(m):
                try:
                    U_new[i, j, :] = np.linalg.lstsq(ATA[i, j], AT1[i, j], rcond=None)[0]
                except Exception:
                    U_new[i, j, :] = U3[i, j, :]
        U3 = U_new
        U = U3.reshape(n * m, r)

        # Update V: for each (j,k): sum_r V[j,k,r]*U[i,j,r]*W[k,i,r] = 1
        # A2[j,k,r,i] = U[i,j,r]*W[k,i,r]; solve A2^T A2 @ V[j,k,:] = A2^T @ 1
        A2 = np.einsum('ijr,kir->jkri', U3, W3, optimize=True)  # (m,p,r,n)
        ATA2 = np.einsum('jkri,jkrs->jkis', A2, A2, optimize=True)
        AT12 = np.einsum('jkri->jkr', A2)
        V_new = np.empty_like(V3)
        for j in range(m):
            for k in range(p):
                try:
                    V_new[j, k, :] = np.linalg.lstsq(ATA2[j, k], AT12[j, k], rcond=None)[0]
                except Exception:
                    V_new[j, k, :] = V3[j, k, :]
        V3 = V_new
        V = V3.reshape(m * p, r)

        # Update W: for each (k,i): sum_r W[k,i,r]*U[i,j,r]*V[j,k,r] = 1
        A3 = np.einsum('ijr,jkr->kirj', U3, V3, optimize=True)  # (p,n,r,m)
        ATA3 = np.einsum('kirj,kirs->kij', A3, A3, optimize=True)
        AT13 = np.einsum('kirj->kir', A3)
        W_new = np.empty_like(W3)
        for k in range(p):
            for i in range(n):
                try:
                    W_new[k, i, :] = np.linalg.lstsq(ATA3[k, i], AT13[k, i], rcond=None)[0]
                except Exception:
                    W_new[k, i, :] = W3[k, i, :]
        W3 = W_new
        W = W3.reshape(p * n, r)

        err = _tensor_error(U, V, W)
        if err < tol:
            break
    return U, V, W, _tensor_error(U, V, W)


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    U, V, W = _build_rank42()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 42
    best_err = _tensor_error(U, V, W)

    # Verify rank-42 solution
    if not _tensor_check(U, V, W, tol=1e-6):
        # Should not happen; return as-is
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try to reduce rank: structured block pruning + ALS polish
    # 6 blocks (2 x 3), each with 7 Strassen columns. Total 42.
    # Try dropping one column from some blocks, then ALS-polish.
    block_cols = []
    for b in range(6):
        block_cols.append(list(range(b * 7, b * 7 + 7)))

    rng = np.random.default_rng(0)
    tried = set()

    for target_rank in range(41, 23, -1):
        if time.time() - start_time > DEADLINE - 15:
            break
        found = False
        n_attempts = 8
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 8:
                break
            # Randomly drop (42 - target_rank) columns from the 42
            drop = rng.choice(42, size=42 - target_rank, replace=False)
            keep = [c for c in range(42) if c not in set(drop.tolist())]
            key = tuple(sorted(keep))
            if key in tried:
                continue
            tried.add(key)
            U_try = U[:, keep].copy()
            V_try = V[:, keep].copy()
            W_try = W[:, keep].copy()
            err0 = _tensor_error(U_try, V_try, W_try)
            if err0 < 1e-9:
                # Already exact
                U_r = _round_to_halfint(U_try)
                V_r = _round_to_halfint(V_try)
                W_r = _round_to_halfint(W_try)
                if U_r is not None and V_r is not None and W_r is not None:
                    if _tensor_check(U_r, V_r, W_r, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
                else:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    found = True
                    break
            # Try ALS polish
            U_p, V_p, W_p, err_p = _als_polish(
                U_try, V_try, W_try, max_iters=40, tol=1e-10,
                time_limit=min(8.0, max(1.0, DEADLINE - 8 - (time.time() - start_time)))
            )
            if err_p < 1e-9:
                U_r = _round_to_halfint(U_p)
                V_r = _round_to_halfint(V_p)
                W_r = _round_to_halfint(W_p)
                if U_r is not None and V_r is not None and W_r is not None:
                    if _tensor_check(U_r, V_r, W_r, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
                else:
                    best_U, best_V, best_W = U_p, V_p, W_p
                    best_rank = target_rank
                    found = True
                    break
        if found:
            continue

    # Final round to half-integers
    U_final = _round_to_halfint(best_U)
    V_final = _round_to_halfint(best_V)
    W_final = _round_to_halfint(best_W)
    if U_final is None or V_final is None or W_final is None:
        U_final, V_final, W_final = best_U, best_V, best_W

    # Final verification; fall back to rank-42 if invalid
    if not _tensor_check(U_final, V_final, W_final, tol=1e-8):
        U_final, V_final, W_final = _build_rank42()

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
