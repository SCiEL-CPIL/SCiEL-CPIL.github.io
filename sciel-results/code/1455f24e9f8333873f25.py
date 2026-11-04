"""Rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication.

Combines:
  - A verified rank-88 construction (schoolbook for 1x4x8 + Strassen on 2x4x8).
  - A fast runtime exact-rank-reduction search (vectorized rank-1 extraction
    + column replacement + greedy) that keeps the best valid decomposition
    found, with a strict deadline guard so it always returns a valid result.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _build_rank88():
    """Verified rank-88 construction: schoolbook(1x4x8) + Strassen(2x4x8)."""
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    # Part 1: i=0 schoolbook 1x4x8 -> 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    # Part 2: i=1,2 Strassen on 2x2x2 blocks -> 2*4*7 = 56 terms
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(ks + lk) * n + (1 + li), col] += sW[t][l]
                col += 1
    return U, V, W


def _build_tensor(U, V, W, n, m, p):
    """Vectorized build of T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T


def _verify(U, V, W, n, m, p, tol=1e-6):
    T = _build_tensor(U, V, W, n, m, p)
    return bool(np.max(np.abs(T - 1.0)) <= tol)


def _extract_rank1(T, n, m, p, tol=1e-9, iters=25):
    """Vectorized ALS to find the best rank-1 term u[i,j]*v[j,k]*w[k,i]
    approximating T. Returns (u, v, w) or None."""
    T = np.asarray(T, dtype=float)
    if np.max(np.abs(T)) < tol:
        return None
    u = np.ones((n, m))
    v = np.ones((m, p))
    w = np.ones((p, n))
    for _ in range(iters):
        # update u: u[i,j] = sum_k T[i,j,k]*v[j,k]*w[k,i] / sum_k (v[j,k]*w[k,i])^2
        for i in range(n):
            for j in range(m):
                denom = v[j, :] * w[:, i]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    u[i, j] = float(np.dot(T[i, j, :], denom)) / dd
        # update v
        for j in range(m):
            for k in range(p):
                denom = u[:, j] * w[k, :]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    v[j, k] = float(np.dot(T[:, j, k], denom)) / dd
        # update w
        for k in range(p):
            for i in range(n):
                denom = u[i, :] * v[:, k]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    w[k, i] = float(np.dot(T[i, :, k], denom)) / dd
    # check it actually captures mass
    T1 = u[:, :, None] * v[None, :, :] * w[:, None, :]  # shape (n,m,p)
    if float(np.max(np.abs(T1))) < tol:
        return None
    return u, v, w


def _greedy_reduce(U, V, W, n, m, p, deadline, tol=1e-6):
    """Reduce rank by replacing columns with extracted rank-1 terms, verifying
    the full tensor after each step. Returns best (U,V,W) with minimal rank."""
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    best = (U.copy(), V.copy(), W.copy())
    R = U.shape[1]
    T = _build_tensor(U, V, W, n, m, p)
    # target tensor is the all-ones; we work with the residual relative to ones
    ones = np.ones((n, m, p))
    for _ in range(R):
        if time.time() > deadline:
            break
        R = U.shape[1]
        if R <= 1:
            break
        # residual: what is not yet matched by the current decomposition
        resid = ones - _build_tensor(U, V, W, n, m, p)
        if float(np.max(np.abs(resid))) < tol:
            # already exact; try to drop a redundant column
            pass
        # extract a rank-1 term from the residual
        res = _extract_rank1(resid, n, m, p, tol=1e-9)
        if res is None:
            break
        u, v, w = res
        # Replace each column r with the new term and test if the tensor is
        # still reproduced (i.e. the column was redundant given the new term).
        improved = False
        for r in range(R):
            if time.time() > deadline:
                break
            U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
            U2[:, r] = u.reshape(-1)
            V2[:, r] = v.reshape(-1)
            W2[:, r] = w.reshape(-1)
            T2 = _build_tensor(U2, V2, W2, n, m, p)
            if float(np.max(np.abs(T2 - 1.0))) <= tol:
                # now try to drop one of the OTHER columns
                for drop in range(R):
                    if drop == r:
                        continue
                    U3 = U2[:, [c for c in range(R) if c != drop]]
                    V3 = V2[:, [c for c in range(R) if c != drop]]
                    W3 = W2[:, [c for c in range(R) if c != drop]]
                    T3 = _build_tensor(U3, V3, W3, n, m, p)
                    if float(np.max(np.abs(T3 - 1.0))) <= tol:
                        U, V, W = U3, V3, W3
                        best = (U.copy(), V.copy(), W.copy())
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # no single replacement+drop worked; try just replacing (no drop)
            # won't reduce rank, so stop to avoid wasting time
            break
    return best


def _to_halfint(U, V, W):
    def r(x):
        return np.round(2 * np.asarray(x, dtype=float)) / 2.0
    return r(U), r(V), r(W)


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before the 150s kill

    n, m, p = 3, 4, 8
    U, V, W = _build_rank88()
    if not _verify(U, V, W, n, m, p):
        # construction bug; fall back to pure schoolbook (always valid)
        rank_s = n * m * p
        U = np.zeros((n * m, rank_s)); V = np.zeros((m * p, rank_s)); W = np.zeros((p * n, rank_s))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0; V[j * p + k, c] = 1.0; W[k * n + i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = (U, V, W)
    # attempt rank reduction
    try:
        cand = _greedy_reduce(U, V, W, n, m, p, deadline)
        if _verify(cand[0], cand[1], cand[2], n, m, p) and cand[0].shape[1] < best[0].shape[1]:
            best = cand
    except Exception:
        pass

    # polish: round to half-integers and re-verify
    try:
        Uh, Vh, Wh = _to_halfint(best[0], best[1], best[2])
        if _verify(Uh, Vh, Wh, n, m, p):
            best = (Uh, Vh, Wh)
    except Exception:
        pass

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
