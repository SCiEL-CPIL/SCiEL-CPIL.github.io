import numpy as np
from scipy.optimize import least_squares
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized exact verification. U:(n*m,r) V:(m*p,r) W:(p*n,r)."""
    rank = U.shape[1]
    # Build index arrays for triples (i,j,k) -> (u,v,w)
    # u = i*m + j ; v = j*p + k ; w = k*n + i
    # Compute T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c] for all triples
    # Vectorize: for each c, outer product U[:,c] @ V[:,c]^T gives (n*m, m*p)
    # Then contract with W along the (v,w) axis.
    # T[u,w] = sum_v sum_c U[u,c] V[v,c] W[w,c]  but we only need specific (u,v,w).
    # Simpler: build full tensor via einsum.
    T = np.einsum('uc,vc,wc->uvw', U, V, W)  # shape (n*m, m*p, p*n)
    # Now check triples
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _prefilter_counts(U, V, W, n, m, p):
    """Cheap necessary condition for ±1 matrices: for each triple,
    sum_c prod must be 1. For ±1 entries, sum_c prod = (agree - disagree)
    = 2*C - rank, where C = # of c where product is +1.
    So 2*C - rank = 1  =>  rank must be odd and C = (rank+1)/2.
    Equivalently rank must be odd. Also sum must equal 1 exactly.
    For general half-integers this doesn't apply, but for pure ±1 it does.
    We use this only as a fast reject for the ±1 random search."""
    rank = U.shape[1]
    if rank % 2 == 0:
        return False
    # For each triple, count sign agreements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                prod = U[u, :] * V[v, :] * W[w, :]
                # prod is ±1
                agree = np.count_nonzero(prod == 1)
                if agree != (rank + 1) // 2:
                    return False
    return True

def _try_random_int(rank, n, m, p, n_attempts, deadline, rng):
    """Random ±1 search with pre-filter."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        U = np.random.choice([-1, 1], size=(size_u, rank))
        V = np.random.choice([-1, 1], size=(size_v, rank))
        W = np.random.choice([-1, 1], size=(size_w, rank))
        # Pre-filter
        if not _prefilter_counts(U, V, W, n, m, p):
            continue
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    return None

def _try_least_squares(rank, n, m, p, n_attempts, deadline, rng):
    """Structured optimization search for half-integer solutions."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = (size_u + size_v + size_w) * rank

    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    num_triples = len(triples)

    def unpack(x):
        U = x[:size_u * rank].reshape(size_u, rank)
        V = x[size_u * rank:(size_u + size_v) * rank].reshape(size_v, rank)
        W = x[(size_u + size_v) * rank:].reshape(size_w, rank)
        return U, V, W

    def residuals(x):
        U, V, W = unpack(x)
        res = np.empty(num_triples)
        for idx, (u, v, w) in enumerate(triples):
            res[idx] = np.dot(U[u, :], V[v, :] * W[w, :]) - 1.0
        return res

    def jacobian(x):
        U, V, W = unpack(x)
        J = np.zeros((num_triples, total_size))
        for idx, (u, v, w) in enumerate(triples):
            VW = V[v, :] * W[w, :]
            UW = U[u, :] * W[w, :]
            UV = U[u, :] * V[v, :]
            J[idx, u * rank:(u + 1) * rank] = VW
            J[idx, size_u * rank + v * rank:size_u * rank + (v + 1) * rank] = UW
            J[idx, (size_u + size_v) * rank + w * rank:(size_u + size_v) * rank + (w + 1) * rank] = UV
        return J

    best = None
    best_norm = np.inf
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
        x0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=total_size)
        try:
            sol = least_squares(residuals, x0, jac=jacobian, method='lm',
                                max_nfev=2000)
            norm = np.linalg.norm(sol.fun)
            if norm < best_norm:
                best_norm = norm
                best = sol.x
            if norm < 1e-7:
                U, V, W = unpack(sol.x)
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
        except Exception:
            continue
    # Polish the best near-solution if it exists
    if best is not None and best_norm < 1e-4:
        try:
            sol = least_squares(residuals, best, jac=jacobian, method='lm',
                                max_nfev=10000)
            U, V, W = unpack(sol.x)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        except Exception:
            pass
    return None

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)

    # Search ranks from low to high, stop at first valid
    for rank in range(17, best_rank):
        if time.time() > deadline - 20:
            break
        # Phase 1: random ±1 with pre-filter (fast)
        result = _try_random_int(rank, n, m, p, n_attempts=5000,
                                 deadline=deadline - 20, rng=None)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
        # Phase 2: structured least-squares
        if time.time() > deadline - 20:
            break
        result = _try_least_squares(rank, n, m, p, n_attempts=30,
                                    deadline=deadline - 20, rng=None)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break

    # Convert to lists of integers/half-integers
    U_out = [[(int(x * 2) / 2) for x in row] for row in best_U]
    V_out = [[(int(x * 2) / 2) for x in row] for row in best_V]
    W_out = [[(int(x * 2) / 2) for x in row] for row in best_W]

    return {"U": U_out, "V": V_out, "W": W_out}
