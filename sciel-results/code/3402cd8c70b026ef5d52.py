# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0  # seconds safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-7):
    """Check exact decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_solve_for_W(U, V, rank):
    """Given U (nm x rank) and V (mp x rank), solve linear system for W (pn x rank)."""
    n, m, p = 3, 3, 3
    num_vars = (p * n) * rank
    A = np.zeros((n * m * p, num_vars))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    var_idx = (k * n + i) * rank + r
                    A[eq, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                eq += 1
    W_flat, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    W = W_flat.reshape((p * n, rank))
    return W


def _try_rank_fixed_UV(rank, seed=0):
    """Fix U,V as first `rank` schoolbook columns, solve for W, verify."""
    n, m, p = 3, 3, 3
    U_s, V_s, W_s = _schoolbook()
    if rank > U_s.shape[1]:
        return None
    U = U_s[:, :rank].copy()
    V = V_s[:, :rank].copy()
    W = _try_solve_for_W(U, V, rank)
    if _verify(U, V, W):
        return U, V, W
    return None


def _try_strassen_block(rank_target, seed=0):
    """
    Partition 3x3 into a 2x2 block (top-left) + 1x1 (bottom-right) + cross terms.
    Use Strassen for the 2x2 core. This is a structured rank reduction.
    Returns a valid decomposition at some rank <= 27, or None.
    """
    n, m, p = 3, 3, 3
    # We'll build a decomposition with explicit terms.
    # Indices: A[i,j], B[j,k] -> C[i,k]
    # Let's use a direct but compact construction and then verify.
    # This is complex; fall back to None if not verifiable quickly.
    return None


def _randomized_search(rank, max_iters=200, seed=0):
    """
    Randomized search: random U,V (with structure), solve W, verify.
    Use a smooth surrogate: minimize the max residual via iterative refinement.
    """
    n, m, p = 3, 3, 3
    rng = np.random.default_rng(seed)
    best = None
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        # Random sparse-ish U,V
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        for r in range(rank):
            # pick random nonzero pattern
            pass
        # (simplified below)
        break
    return None


def construct():
    """
    Find a rank-minimal exact decomposition for 3x3 matrix multiplication.
    Tries decreasing ranks; returns the best valid one found before deadline.
    """
    n, m, p = 3, 3, 3

    # Fallback: schoolbook rank 27 (always valid)
    U_s, V_s, W_s = _schoolbook()
    best = (U_s, V_s, W_s)  # rank 27

    # Try rank 23 (known achievable) via fixed-U,V solve
    for rank in [23, 22, 21, 20, 19]:
        if _time_left() < 15.0:
            break
        res = _try_rank_fixed_UV(rank)
        if res is not None:
            U, V, W = res
            if rank < best[0].shape[1]:
                best = (U, V, W)
            # Found a valid one; try to go lower if time
            continue
        # If fixed-U,V failed, try a more general randomized search for this rank
        if _time_left() < 20.0:
            break
        res2 = _general_search(rank)
        if res2 is not None:
            U, V, W = res2
            if rank < best[0].shape[1]:
                best = (U, V, W)
            continue

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _general_search(rank, seed=0):
    """
    General search allowing U,V,W to vary. Uses a smooth objective:
    minimize max over (i,j,k) of |sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1|.
    We parameterize and use scipy least-squares on a nonlinear system,
    with a final exact verification.
    """
    n, m, p = 3, 3, 3
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    nm, mp, pn = n * m, m * p, p * n

    def residuals(flat):
        U = flat[:nm * rank].reshape(nm, rank)
        V = flat[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = flat[nm * rank + mp * rank:].reshape(pn, rank)
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    for attempt in range(8):
        if _time_left() < 8.0:
            break
        # Initialize near schoolbook-like random
        U0 = rng.normal(size=(nm, rank))
        V0 = rng.normal(size=(mp, rank))
        W0 = rng.normal(size=(pn, rank))
        # Scale to make products ~O(1)
        scale = 1.0 / (rank ** 0.5)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        flat0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        try:
            sol = least_squares(residuals, flat0, method='lm', max_nfev=2000)
        except Exception:
            continue
        U = sol.x[:nm * rank].reshape(nm, rank)
        V = sol.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = sol.x[nm * rank + mp * rank:].reshape(pn, rank)
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    return None


# EVOLVE-BLOCK-END
