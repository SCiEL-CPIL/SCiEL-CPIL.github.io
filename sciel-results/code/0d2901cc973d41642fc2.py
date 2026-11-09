"""Search for a rank-minimal exact tensor decomposition of 4x4x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 4]
_DEADLINE = time.time() + 115.0  # safety margin before the 150s kill


def _build_schoolbook():
    n, m, p = 4, 4, 4
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _check(U, V, W, tol=1e-9):
    """Check sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k."""
    n, m, p = 4, 4, 4
    r = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err <= tol, max_err


def _random_init(r, seed):
    rng = random.Random(seed)
    n, m, p = 4, 4, 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for idx in range(n * m):
        for rr in range(r):
            U[idx, rr] = rng.uniform(-1, 1)
    for idx in range(m * p):
        for rr in range(r):
            V[idx, rr] = rng.uniform(-1, 1)
    for idx in range(p * n):
        for rr in range(r):
            W[idx, rr] = rng.uniform(-1, 1)
    return U, V, W


def _refine(U, V, W, iters=400):
    """Alternating least squares to satisfy sum_r U[i,j,r]V[j,k,r]W[k,i,r] = 1."""
    n, m, p = 4, 4, 4
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    for it in range(iters):
        # Update U[i,j,:] by solving G[i,j] @ U[i,j,:] = ones(p)
        for i in range(n):
            for j in range(m):
                G = np.zeros((p, r))
                for k in range(p):
                    G[k, :] = V[j * p + k, :] * W[k * n + i, :]
                if r >= p:
                    x, *_ = np.linalg.lstsq(G, np.ones(p), rcond=None)
                else:
                    x = np.ones(r)
                U[i * m + j, :] = x
        # Update V[j,k,:] by solving H[j,k] @ V[j,k,:] = ones(m)
        for j in range(m):
            for k in range(p):
                H = np.zeros((m, r))
                for i in range(n):
                    H[i, :] = U[i * m + j, :] * W[k * n + i, :]
                if r >= m:
                    x, *_ = np.linalg.lstsq(H, np.ones(m), rcond=None)
                else:
                    x = np.ones(r)
                V[j * p + k, :] = x
        # Update W[k,i,:] by solving K[k,i] @ W[k,i,:] = ones(n)
        for k in range(p):
            for i in range(n):
                K = np.zeros((n, r))
                for j in range(m):
                    K[j, :] = U[i * m + j, :] * V[j * p + k, :]
                if r >= n:
                    x, *_ = np.linalg.lstsq(K, np.ones(n), rcond=None)
                else:
                    x = np.ones(r)
                W[k * n + i, :] = x
        # Check convergence
        ok, err = _check(U, V, W)
        if ok:
            break
    return U, V, W, err


def construct():
    t0 = time.time()
    n, m, p = 4, 4, 4
    # Fallback: schoolbook
    U0, V0, W0 = _build_schoolbook()
    best = (U0, V0, W0)
    best_rank = 64

    # Try decreasing ranks; best known upper bound for 4x4x4 is 29
    for r in range(29, 15, -1):
        if time.time() > _DEADLINE - 20:
            break
        found = False
        for seed in range(0, 10):
            if time.time() > _DEADLINE - 15:
                break
            U, V, W = _random_init(r, seed)
            U, V, W, err = _refine(U, V, W, iters=400)
            ok, err = _check(U, V, W)
            if ok:
                best = (U, V, W)
                best_rank = r
                found = True
                break
        if found:
            continue  # try lower rank
        else:
            break  # couldn't find at this rank; lower is harder

    U, V, W = best
    r = U.shape[1]
    U_list = [[float(U[i, j]) for j in range(r)] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(r)] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(r)] for i in range(W.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}
