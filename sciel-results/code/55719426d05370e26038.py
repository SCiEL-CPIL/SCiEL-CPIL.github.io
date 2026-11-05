import time
import numpy as np

DIMENSIONS = [3, 5, 7]
START = time.time()
DEADLINE = 150.0
MARGIN = 25.0

def _time_left():
    return DEADLINE - MARGIN - (time.time() - START)

def _verify(U, V, W, n, m, p, tol=1e-7):
    r = len(U[0])
    A = np.array(U, dtype=float).reshape(n, m, r)
    B = np.array(V, dtype=float).reshape(m, p, r)
    C = np.array(W, dtype=float).reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', A, B, C)
    target = np.ones((n, m, p))
    return bool(np.max(np.abs(T - target)) < tol)

def _to_halfint(mat, tol=1e-7):
    m = np.array(mat, dtype=float)
    snapped = np.round(m * 2.0) / 2.0
    if np.max(np.abs(m - snapped)) > tol:
        return False, None
    return True, snapped

def _als_decompose(T, n, m, p, tr, iters=400, tol=1e-9, seed=0):
    rng = np.random.default_rng(seed)
    Umat = rng.standard_normal((n * m, tr))
    Vmat = rng.standard_normal((m * p, tr))
    Wmat = rng.standard_normal((p * n, tr))
    # normalize columns of U
    nrm = np.linalg.norm(Umat, axis=0); nrm[nrm < 1e-12] = 1.0
    Umat = Umat / nrm
    best_err = np.inf
    best = (Umat.copy(), Vmat.copy(), Wmat.copy())
    for it in range(iters):
        if _time_left() < 5:
            break
        for i in range(n):
            for j in range(m):
                Mu = np.zeros((p, tr))
                for k in range(p):
                    Mu[k, :] = Vmat[j * p + k, :] * Wmat[k * n + i, :]
                Umat[i * m + j, :] = np.linalg.lstsq(Mu.T, T[i, j, :], rcond=None)[0]
        for j in range(m):
            for k in range(p):
                Mv = np.zeros((n, tr))
                for i in range(n):
                    Mv[i, :] = Umat[i * m + j, :] * Wmat[k * n + i, :]
                Vmat[j * p + k, :] = np.linalg.lstsq(Mv.T, T[:, j, k], rcond=None)[0]
        for k in range(p):
            for i in range(n):
                Mw = np.zeros((m, tr))
                for j in range(m):
                    Mw[j, :] = Umat[i * m + j, :] * Vmat[j * p + k, :]
                Wmat[k * n + i, :] = np.linalg.lstsq(Mw.T, T[i, :, k], rcond=None)[0]
        A_ = Umat.reshape(n, m, tr)
        B_ = Vmat.reshape(m, p, tr)
        C_ = Wmat.reshape(p, n, tr)
        Trec = np.einsum('ijc,jkc,kic->ijk', A_, B_, C_)
        err = float(np.linalg.norm(Trec - T))
        if err < best_err:
            best_err = err
            best = (Umat.copy(), Vmat.copy(), Wmat.copy())
        if err < tol:
            break
    Umat, Vmat, Wmat = best
    A_ = Umat.reshape(n, m, tr)
    B_ = Vmat.reshape(m, p, tr)
    C_ = Wmat.reshape(p, n, tr)
    Trec = np.einsum('ijc,jkc,kic->ijk', A_, B_, C_)
    if float(np.max(np.abs(Trec - T))) > 1e-5:
        return None
    return Umat, Vmat, Wmat

def _schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r)); V = np.zeros((m * p, r)); W = np.zeros((p * n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _try_rank(U, V, W, n, m, p, tr, seed=0):
    A_ = np.array(U, float).reshape(n, m, U.shape[1])
    B_ = np.array(V, float).reshape(m, p, V.shape[1])
    C_ = np.array(W, float).reshape(p, n, W.shape[1])
    T = np.einsum('ijc,jkc,kic->ijk', A_, B_, C_)
    res = _als_decompose(T, n, m, p, tr, seed=seed)
    if res is None:
        return None
    Umat, Vmat, Wmat = res
    candidates = []
    for scale in [1.0, 0.5, 2.0, 0.25, 4.0]:
        U2 = Umat * scale
        V2 = Vmat / scale
        W2 = Wmat
        okU, sU = _to_halfint(U2)
        okV, sV = _to_halfint(V2)
        okW, sW = _to_halfint(W2)
        if okU and okV and okW:
            if _verify(sU.tolist(), sV.tolist(), sW.tolist(), n, m, p, tol=1e-6):
                return (sU, sV, sW)
    return None

def construct():
    n, m, p = 3, 5, 7
    U, V, W = _schoolbook(n, m, p)
    # Fallback: schoolbook is always valid
    fallback = (U.copy(), V.copy(), W.copy())
    best_rank = n * m * p
    best = (fallback[0], fallback[1], fallback[2])
    # Try to reduce rank, high to low, stop at 99
    for tr in range(104, 98, -1):
        if _time_left() < 8:
            break
        found = None
        for seed in range(3):
            if _time_left() < 8:
                break
            found = _try_rank(U, V, W, n, m, p, tr, seed=seed)
            if found is not None:
                break
        if found is not None:
            sU, sV, sW = found
            if _verify(sU.tolist(), sV.tolist(), sW.tolist(), n, m, p, tol=1e-6):
                best_rank = tr
                best = (sU, sV, sW)
                break
    sU, sV, sW = best
    return {"U": sU.tolist(), "V": sV.tolist(), "W": sW.tolist()}
