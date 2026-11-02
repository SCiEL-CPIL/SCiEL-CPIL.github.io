import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START = time.time()
DEADLINE = _START + 140.0

def _remaining():
    return DEADLINE - time.time()

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _recon(U, V, W):
    return np.einsum('al,bl,cl->abc', U, V, W)

def _res(U, V, W, T):
    return np.linalg.norm(_recon(U, V, W) - T)

def _als(U, V, W, T, iters=300, tol=1e-13):
    r = U.shape[1]
    eye = np.eye(r)
    prev = _res(U, V, W, T)
    for it in range(iters):
        if _remaining() < 8: break
        # U
        G = (V.T @ V) * (W.T @ W)
        G = G + 1e-10 * (np.trace(G) / r) * eye
        rhs = np.einsum('acl,cl->al', np.einsum('abc,bl->acl', T, V), W)
        U = np.linalg.solve(G, rhs.T).T
        # V
        G = (U.T @ U) * (W.T @ W)
        G = G + 1e-10 * (np.trace(G) / r) * eye
        rhs = np.einsum('abl,al->bl', np.einsum('abc,cl->abl', T, W), U)
        V = np.linalg.solve(G, rhs.T).T
        # W
        G = (U.T @ U) * (V.T @ V)
        G = G + 1e-10 * (np.trace(G) / r) * eye
        rhs = np.einsum('bcl,bl->cl', np.einsum('abc,al->bcl', T, U), V)
        W = np.linalg.solve(G, rhs.T).T
        cur = _res(U, V, W, T)
        if abs(prev - cur) < tol:
            prev = cur
            break
        prev = cur
    return U, V, W, cur

def _snap_half(U, V, W, T, tol=1e-7):
    # try rounding to half-integers; return (snapped, ok)
    Us = np.round(U * 2) / 2
    Vs = np.round(V * 2) / 2
    Ws = np.round(W * 2) / 2
    if _res(Us, Vs, Ws, T) < tol:
        return Us, Vs, Ws, True
    return None, None, None, False

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    best = None  # (rank, res, U, V, W)
    # candidate ranks
    ranks = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 48, 52, 56, 59]
    rng = np.random.default_rng(0)
    for r in ranks:
        if _remaining() < 12: break
        # more restarts for smaller r
        nrestart = max(3, min(20, int(3000 / r)))
        found = False
        for it in range(nrestart):
            if _remaining() < 8: break
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            U, V, W, res = _als(U, V, W, T, iters=400)
            if res < 1e-7:
                found = True
                if best is None or r < best[0]:
                    best = (r, res, U, V, W)
                break  # found at this rank, move to next (lower) rank
        if found:
            # we found a solution at rank r; since we go low->high, this is the min found so far
            # but maybe a lower rank also works; continue to check lower? we already checked lower.
            # Actually we go ascending, so first found is the minimum. But to be safe, keep searching a bit.
            # Let's record and continue to ensure we don't miss a lower one (we already tried lower).
            pass
    # pick best
    if best is not None:
        r, res, U, V, W = best
        # try to snap to half-integers
        Us, Vs, Ws, ok = _snap_half(U, V, W, T, tol=1e-6)
        if ok:
            U, V, W = Us, Vs, Ws
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return _schoolbook(n, m, p)
