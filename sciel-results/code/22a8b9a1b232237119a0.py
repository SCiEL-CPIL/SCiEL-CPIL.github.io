import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
n, m, p = DIMENSIONS

_DEADLINE = time.time() + 110.0

def _time_left():
    return _DEADLINE - time.time()

def _build_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T):
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, T, atol=1e-9, rtol=0)

def _snap(arr, tol=1e-8):
    d = np.round(arr * 2)
    if np.allclose(arr * 2, d, atol=tol, rtol=0):
        return d / 2.0
    return None

def _als(T, rank, max_iters=500, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3

    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        for r in range(rank):
            # U[:,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r]
            tmp = np.einsum('abc,c->ab', T, W[:, r])
            U[:, r] = tmp @ V[:, r]
        for r in range(rank):
            tmp = np.einsum('abc,a->bc', T, U[:, r])
            V[:, r] = tmp @ W[:, r]
        for r in range(rank):
            tmp = np.einsum('abc,b->ac', T, V[:, r])
            W[:, r] = tmp @ U[:, r]

        if it % 20 == 0:
            res = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(res - T)
            if err < 1e-10:
                break

    Us = _snap(U)
    Vs = _snap(V)
    Ws = _snap(W)
    if Us is not None and Vs is not None and Ws is not None and _verify(Us, Vs, Ws, T):
        return Us, Vs, Ws
    if _verify(U, V, W, T):
        return U, V, W
    return None

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 110.0

    T = _build_tensor()
    sb_rank = n * m * p

    # Schoolbook
    U_sb = np.zeros((n * m, sb_rank))
    V_sb = np.zeros((m * p, sb_rank))
    W_sb = np.zeros((p * n, sb_rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, c] = 1
                V_sb[j*p+k, c] = 1
                W_sb[k*n+i, c] = 1
                c += 1

    best_rank = sb_rank
    best = (U_sb, V_sb, W_sb)

    # Try ALS for lower ranks
    for rank in range(24, sb_rank):
        if _time_left() < 8.0:
            break
        for seed in range(3):
            if _time_left() < 3.0:
                break
            res = _als(T, rank, max_iters=200, seed=seed)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W, T):
                    best = (U, V, W)
                    best_rank = rank
                    break

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
