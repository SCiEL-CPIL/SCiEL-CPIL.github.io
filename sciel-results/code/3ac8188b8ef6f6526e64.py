import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import sys

_T0 = time.time()
_DEADLINE = _T0 + 120.0   # evaluator wall-clock limit; keep a safety margin
_MARGIN = 6.0             # stop searching this many seconds before the limit


def _time_left():
    return _DEADLINE - _MARGIN - time.time()


def _check_exact(U, V, W, n, m, p):
    """Return (valid, err): whether T = U V^T W^T exactly equals the MM tensor."""
    n, m, p = n, m, p
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T = U @ V.T @ W.T
    # Build reference tensor T_ref[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    Tref = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tref[i * m + j, j * p + k, k * n + i] = 1.0
    err = float(np.linalg.norm(T - Tref))
    valid = err < 1e-4
    return valid, err


def _snap(U, V, W):
    """Round to nearest half-integer and re-verify exactness; return snapped if exact."""
    Us = np.round(U * 2.0) / 2.0
    Vs = np.round(V * 2.0) / 2.0
    Ws = np.round(W * 2.0) / 2.0
    valid, err = _check_exact(Us, Vs, Ws, *DIMENSIONS)
    if valid:
        return Us, Vs, Ws
    return None


def _als_step(U, V, W, T, n, m, p):
    """One alternating-least-squares sweep: solve each factor given the others."""
    # Solve U: U @ (V.T @ W.T) = T  -> U = T @ (V.T @ W.T)^+
    # Equivalently minimize || U A - T ||_F where A = V.T @ W.T
    A = V.T @ W.T
    U = np.linalg.lstsq(A.T, T.T, rcond=None)[0].T
    # Solve V
    B = U @ W.T
    V = np.linalg.lstsq(B.T, T.T, rcond=None)[0].T
    # Solve W
    C = U @ V.T
    W = np.linalg.lstsq(C.T, T.T, rcond=None)[0].T
    return U, V, W


def _rank_of(U, V, W):
    # Number of rank-1 factors actually used (columns not all-zero)
    r = 0
    for M in (U, V, W):
        nz = np.linalg.norm(M, axis=0) > 1e-9
        r = max(r, int(np.sum(nz)))
    return r


def _try_decomposition(n, m, p, T, rank, seed, max_iters):
    rng = np.random.default_rng(seed)
    # Random sparse initialization: each column picks one (i,j), one (j,k), one (k,i)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c in range(rank):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        U[i * m + j, c] = rng.choice([-1.0, 1.0])
        V[j * p + k, c] = rng.choice([-1.0, 1.0])
        W[k * n + i, c] = rng.choice([-1.0, 1.0])

    best_err = float("inf")
    best = None
    prev_err = float("inf")
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        U, V, W = _als_step(U, V, W, T, n, m, p)
        err = _check_exact(U, V, W, n, m, p)[1]
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if best_err < 1e-6:
            # converged; try to snap to exact
            snapped = _snap(U, V, W)
            if snapped is not None:
                return snapped
            # try a few more polish iterations then snap
            for _ in range(20):
                U, V, W = _als_step(U, V, W, T, n, m, p)
            snapped = _snap(U, V, W)
            if snapped is not None:
                return snapped
            break
        if abs(prev_err - err) < 1e-9:
            prev_err = err
        else:
            prev_err = err
    # final snap attempt on best
    if best is not None:
        snapped = _snap(*best)
        if snapped is not None:
            return snapped
    return None


def construct():
    n, m, p = DIMENSIONS
    # Reference tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    best = None
    best_rank = None

    # Search over candidate ranks in increasing order; for each, many random restarts.
    # Lower bound on rank is max(n*p, m*p, n*m) = max(24, 32, 12) = 32.
    lo = max(n * p, m * p, n * m)
    hi = n * m * p  # schoolbook
    # We know the true rank is well below schoolbook; probe a grid.
    candidate_ranks = []
    r = lo
    while r <= hi:
        candidate_ranks.append(r)
        r += 2
    if hi not in candidate_ranks:
        candidate_ranks.append(hi)

    seed = 0
    for rank in candidate_ranks:
        if _time_left() < 8.0:
            break
        # number of restarts for this rank: more for lower ranks (harder to hit)
        restarts = 40
        for _ in range(restarts):
            if _time_left() < 4.0:
                break
            seed += 1
            res = _try_decomposition(n, m, p, T, rank, seed, max_iters=200)
            if res is not None:
                U, V, W = res
                valid, err = _check_exact(U, V, W, n, m, p)
                if valid:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best = (U, V, W)
                    # found a valid one at this rank; stop searching higher ranks
                    break
        if best_rank is not None and best_rank <= rank:
            break

    # Fallback: schoolbook if nothing found (should not happen)
    if best is None:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best = (U, V, W)

    U, V, W = best
    # Convert to lists for the return interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
