import time
import random
import copy
import numpy as np

DIMENSIONS = [2, 4, 8]
DEADLINE = 150.0
MARGIN = 12.0
RNG_SEED = 12345

def _time_left():
    return DEADLINE - MARGIN - time.time()

def _validate(U, V, W, n, m, p):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Build multiplication tensor T[i,j,k] and check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                acc = float(np.dot(uc, vc) * 0)  # placeholder
                # elementwise product then sum
                acc = float(np.sum(uc * vc * wc))
                if abs(acc - 1.0) > 1e-6:
                    return False
    return True

def _score_rank(U, V, W):
    return U.shape[1]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _to_list(M):
    return M.tolist()

def construct():
    start = time.time()
    random.seed(RNG_SEED)
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best = (U.copy(), V.copy(), W.copy())
    best_rank = best[0].shape[1]

    # Local search: try to drop a random column by solving for its replacement in the span of others.
    # If a column can be expressed as a linear combination of the others in U, V, W simultaneously,
    # we can drop it. We use a randomized approach: pick a column c, and try to find coefficients
    # on the remaining columns that reproduce U[:,c], V[:,c], W[:,c].
    # This is a linear system; solve least squares and check residual.

    def try_drop_column(U, V, W, c, rng):
        rank = U.shape[1]
        if rank <= 1:
            return None
        keep = [i for i in range(rank) if i != c]
        Uk = U[:, keep]
        Vk = V[:, keep]
        Wk = W[:, keep]
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        # Solve for x such that Uk @ x ~ Uc, Vk @ x ~ Vc, Wk @ x ~ Wc
        # Use combined least squares
        A = np.vstack([Uk, Vk, Wk])
        b = np.concatenate([Uc, Vc, Wc])
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        # Check residual
        resid = np.linalg.norm(A @ x - b)
        if resid < 1e-7:
            # Drop column c
            newU = np.delete(U, c, axis=1)
            newV = np.delete(V, c, axis=1)
            newW = np.delete(W, c, axis=1)
            return newU, newV, newW
        return None

    # Also try merging: replace two columns by their sum (if that preserves the tensor).
    # Actually, the linear combination approach above is more general.

    # Iterative: repeatedly try to drop a random column.
    iter_count = 0
    while _time_left() > 0.5:
        iter_count += 1
        rank = U.shape[1]
        if rank <= 1:
            break
        # Try several random columns
        found = False
        for attempt in range(20):
            if _time_left() < 0.5:
                break
            c = rng.randint(0, rank - 1)
            result = try_drop_column(U, V, W, c, rng)
            if result is not None:
                U, V, W = result
                found = True
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best = (U.copy(), V.copy(), W.copy())
                break
        if not found:
            # Try a few more columns from a different random seed
            for attempt in range(10):
                if _time_left() < 0.5:
                    break
                c = rng.randint(0, rank - 1)
                result = try_drop_column(U, V, W, c, rng)
                if result is not None:
                    U, V, W = result
                    found = True
                    if U.shape[1] < best_rank:
                        best_rank = U.shape[1]
                        best = (U.copy(), V.copy(), W.copy())
                    break
        if not found and iter_count > 50:
            # Stuck; try a randomized restart with perturbation
            # Perturb: add small random noise to a random column, then re-optimize.
            # This may escape local minima.
            if _time_left() > 1.0:
                c = rng.randint(0, U.shape[1] - 1)
                noise = 0.01 * rng.random()
                U[:, c] += noise
                V[:, c] += noise
                W[:, c] += noise
                # Re-check validity; if invalid, revert
                if not _validate(U, V, W, n, m, p):
                    U, V, W = best[0].copy(), best[1].copy(), best[2].copy()
                else:
                    # Continue search
                    pass

    # Final validation of best
    if not _validate(best[0], best[1], best[2], n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best = (U.copy(), V.copy(), W.copy())

    return {"U": _to_list(best[0]), "V": _to_list(best[1]), "W": _to_list(best[2])}
