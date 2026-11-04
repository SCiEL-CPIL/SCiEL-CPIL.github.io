import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]


def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Return True iff sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for every
    (i,j,k) and 0 elsewhere."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-8:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-8:
        return False
    return True


def _fold_once(U, V, W, n, m, p, rng):
    """Attempt one rank-reducing fold: pick a term to remove, add a random rank-1
    tensor to every remaining term, then drop the removed term.  Returns new
    (U,V,W) or None if no valid random direction was found quickly."""
    r = len(U[0])
    if r <= 1:
        return None
    # pick term to remove (prefer random; keep first term as anchor sometimes)
    if rng.random() < 0.7:
        remove = rng.randrange(r)
    else:
        remove = 0
    # build random rank-1 tensor (a,b,c)
    a = rng.gauss(0.0, 1.0) * np.ones(n * m)
    b = rng.gauss(0.0, 1.0) * np.ones(m * p)
    c = rng.gauss(0.0, 1.0) * np.ones(p * n)
    # scale to make the fold numerically stable (small perturbation)
    scale = 0.5
    a = a * scale
    b = b * scale
    c = c * scale
    new_U = [row[:] for row in U]
    new_V = [row[:] for row in V]
    new_W = [row[:] for row in W]
    # add rank-1 tensor to every term
    for t in range(r):
        if t == remove:
            continue
        for i in range(n * m):
            new_U[i][t] += a[i]
        for j in range(m * p):
            new_V[j][t] += b[j]
        for k in range(p * n):
            new_W[k][t] += c[k]
    # drop the removed term
    new_U = [row[:remove] + row[remove + 1:] for row in new_U]
    new_V = [row[:remove] + row[remove + 1:] for row in new_V]
    new_W = [row[:remove] + row[remove + 1:] for row in new_W]
    return new_U, new_V, new_W


def _search_lower_rank(U, V, W, n, m, p, deadline, seed):
    """Local search: repeatedly attempt folds; keep best valid decomposition."""
    rng = random.Random(seed)
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    start = time.time()
    attempts = 0
    while time.time() < deadline - 5.0 and best_rank > 1:
        attempts += 1
        candidate = _fold_once(best_U, best_V, best_W, n, m, p, rng)
        if candidate is None:
            continue
        cU, cV, cW = candidate
        # quick numerical check via einsum (fast)
        Ua = np.asarray(cU, dtype=float)
        Va = np.asarray(cV, dtype=float)
        Wa = np.asarray(cW, dtype=float)
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-7:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            T2 = T.copy()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T2[i * m + j, j * p + k, k * n + i] = 0.0
            if np.max(np.abs(T2)) > 1e-7:
                ok = False
        if ok:
            best_U, best_V, best_W = cU, cV, cW
            best_rank = len(cU[0])
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin under 150 s

    # Start from schoolbook (guaranteed valid, rank 144).
    res = _build_schoolbook(n, m, p)
    U, V, W = res["U"], res["V"], res["W"]

    # Try to reduce rank via local search (fold moves).
    U2, V2, W2, rank2 = _search_lower_rank(U, V, W, n, m, p, deadline, seed=12345)
    if rank2 < len(U[0]):
        U, V, W = U2, V2, W2

    # Final verification before returning.
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: guaranteed schoolbook.
    return _build_schoolbook(n, m, p)
