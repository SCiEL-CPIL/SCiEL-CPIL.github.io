"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _triples(n, m, p):
    """List of (u_index, v_index, w_index) for each (i,j,k)."""
    tr = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tr.append((u, v, w))
    return tr

def _residual(U, V, W, tr):
    """Max abs residual over all triples of (sum_r U[u,r]V[v,r]W[w,r] - 1)."""
    r = U.shape[1]
    max_err = 0.0
    for (u, v, w) in tr:
        s = 0.0
        ur = U[u]
        vr = V[v]
        wr = W[w]
        for t in range(r):
            s += ur[t] * vr[t] * wr[t]
        e = abs(s - 1.0)
        if e > max_err:
            max_err = e
    return max_err

def _als_step(U, V, W, tr, n, m, p):
    """One ALS iteration: fix two, least-squares solve the third."""
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n
    # Precompute for efficiency: group triples by u, by v, by w.
    # For solving U: for each u, we need sum over triples with that u of
    #   (sum_r U[u,r] * A_vw[r]) where A_vw[r] = V[v,r]*W[w,r]
    # target: 1 per triple.
    # This is a linear system in U[u, :] : for each triple (u,v,w):
    #   sum_r U[u,r] * (V[v,r]*W[w,r]) = 1
    # Let M_u be a (n_u x r) matrix where each row is (V[v,r]*W[w,r]) for a triple with this u.
    # Then M_u @ U[u,:] = ones(n_u).
    # Solve via least squares.
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    # Solve U
    u_map = {}
    for (u, v, w) in tr:
        u_map.setdefault(u, []).append((v, w))
    for u, lst in u_map.items():
        A = np.zeros((len(lst), r))
        for idx, (v, w) in enumerate(lst):
            A[idx] = V[v] * W[w]
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        U_new[u] = sol
    # Solve V
    v_map = {}
    for (u, v, w) in tr:
        v_map.setdefault(v, []).append((u, w))
    for v, lst in v_map.items():
        A = np.zeros((len(lst), r))
        for idx, (u, w) in enumerate(lst):
            A[idx] = U_new[u] * W[w]
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        V_new[v] = sol
    # Solve W
    w_map = {}
    for (u, v, w) in tr:
        w_map.setdefault(w, []).append((u, v))
    for w, lst in w_map.items():
        A = np.zeros((len(lst), r))
        for idx, (u, v) in enumerate(lst):
            A[idx] = U_new[u] * V_new[v]
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        W_new[w] = sol
    return U_new, V_new, W_new

def _als_search(r, n, m, p, seed, iters, time_limit, start_time):
    tr = _triples(n, m, p)
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = np.ones((p * n, r))  # init W to ones helps symmetry
    best = (float('inf'), U.copy(), V.copy(), W.copy())
    for it in range(iters):
        if time.time() - start_time > time_limit:
            break
        U, V, W = _als_step(U, V, W, tr, n, m, p)
        err = _residual(U, V, W, tr)
        if err < best[0]:
            best = (err, U.copy(), V.copy(), W.copy())
        if err < 1e-9:
            break
    return best

def _to_halfint_list(A):
    """Convert numpy array to list of lists with integer/half-integer entries (rounded)."""
    out = []
    for row in A:
        out.append([float(x) for x in row])
    return out

def construct():
    n, m, p = DIMENSIONS  # 4,5,5
    start_time = time.time()
    HARD_LIMIT = 150.0
    SAFETY = 10.0  # return before limit
    tr = _triples(n, m, p)

    # Start with schoolbook as guaranteed fallback.
    sU, sV, sW = _schoolbook(n, m, p)

    # Try a range of ranks below schoolbook, descending from a candidate.
    # Use the structural hint: rank should be at least n (4) but we search.
    best = (float('inf'), sU, sV, sW)
    # candidate ranks to try
    candidate_ranks = [60, 50, 45, 40, 35, 30, 25, 20, 15]
    seeds = [42, 1, 7, 123]
    iters = 400
    for r in candidate_ranks:
        if time.time() - start_time > HARD_LIMIT - SAFETY:
            break
        for seed in seeds:
            if time.time() - start_time > HARD_LIMIT - SAFETY:
                break
            tl = HARD_LIMIT - SAFETY - (time.time() - start_time)
            if tl < 2:
                break
            err, U, V, W = _als_search(r, n, m, p, seed, iters, tl, start_time)
            if err < best[0]:
                best = (err, U, V, W)
            if best[0] < 1e-9:
                break
        if best[0] < 1e-9:
            break

    # If we found an exact (or near-exact) low-rank solution, use it.
    if best[0] < 1e-9:
        U, V, W = best[1], best[2], best[3]
        # Round to nearest half-integer to satisfy integer/half-integer requirement.
        U = np.round(U * 2.0) / 2.0
        V = np.round(V * 2.0) / 2.0
        W = np.round(W * 2.0) / 2.0
        # Re-verify after rounding; if it broke, fall back to schoolbook.
        if _residual(U, V, W, tr) < 1e-6:
            return {"U": _to_halfint_list(U), "V": _to_halfint_list(V), "W": _to_halfint_list(W)}
    # Fallback: schoolbook (rank 100), exact.
    return {"U": _to_halfint_list(sU), "V": _to_halfint_list(sV), "W": _to_halfint_list(sW)}
# EVOLVE-BLOCK-END
