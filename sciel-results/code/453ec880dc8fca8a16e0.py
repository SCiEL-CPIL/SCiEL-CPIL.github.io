"""Rank-minimal exact tensor decomposition for 2x2 @ 2x3 matrix multiplication."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    r = len(U[0])
    # Coefficient of A[i,j]*B[j,k] in C[i,kp] must be delta(k,kp)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    s = 0.0
                    for ridx in range(r):
                        s += U[i*m+j][ridx] * V[j*p+k][ridx] * W[kp*n+i][ridx]
                    target = 1.0 if kp == k else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _round_if_close(M, tol=1e-6):
    M = np.array(M, dtype=float)
    R = np.rint(M)
    if np.all(np.abs(M - R) < tol):
        return R.astype(int).tolist()
    H = np.rint(2.0 * M) / 2.0
    if np.all(np.abs(M - H) < tol):
        return H.tolist()
    return None

def _search_rank(r, time_budget=3.0, seed=0):
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    t0 = time.time()
    rowsU, rowsV, rowsW = n*m, m*p, p*n
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    eqs.append((i*m+j, j*p+k, kp*n+i, 1.0 if kp == k else 0.0))

    def err_of(U, V, W):
        e = 0.0
        for (iU, iV, iW, tgt) in eqs:
            v = float(np.dot(np.dot(U[iU], V[iV]), W[iW]) - tgt)
            e += v * v
        return e

    def solve_U(V, W):
        A = np.zeros((len(eqs), rowsU * r)); b = np.zeros(len(eqs))
        for e, (iU, iV, iW, tgt) in enumerate(eqs):
            A[e, iU*r:(iU+1)*r] = np.outer(V[iV], W[iW]).reshape(-1)
            b[e] = tgt
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol.reshape(rowsU, r)

    def solve_V(U, W):
        A = np.zeros((len(eqs), rowsV * r)); b = np.zeros(len(eqs))
        for e, (iU, iV, iW, tgt) in enumerate(eqs):
            A[e, iV*r:(iV+1)*r] = np.outer(U[iU], W[iW]).reshape(-1)
            b[e] = tgt
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol.reshape(rowsV, r)

    def solve_W(U, V):
        A = np.zeros((len(eqs), rowsW * r)); b = np.zeros(len(eqs))
        for e, (iU, iV, iW, tgt) in enumerate(eqs):
            A[e, iW*r:(iW+1)*r] = np.outer(U[iU], V[iV]).reshape(-1)
            b[e] = tgt
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol.reshape(rowsW, r)

    best = None
    for attempt in range(80):
        if time.time() - t0 > time_budget:
            break
        U = np.random.randn(rowsU, r); V = np.random.randn(rowsV, r); W = np.random.randn(rowsW, r)
        for it in range(80):
            if time.time() - t0 > time_budget:
                break
            U = solve_U(V, W); V = solve_V(U, W); W = solve_W(U, V)
            e = err_of(U, V, W)
            if e < 1e-16:
                break
            if best is None or e < best[0]:
                best = (e, U.copy(), V.copy(), W.copy())
        e = err_of(U, V, W)
        if best is None or e < best[0]:
            best = (e, U.copy(), V.copy(), W.copy())
        if e < 1e-16:
            break
    if best is None:
        return None
    e, U, V, W = best
    if e > 1e-8:
        return None
    RU, RV, RW = _round_if_close(U), _round_if_close(V), _round_if_close(W)
    if RU is not None and RV is not None and RW is not None:
        if _verify(RU, RV, RW):
            return RU, RV, RW
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1; V[j*p+k][c] = 1; W[k*n+i][c] = 1; c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s kill
    for seed in range(0, 400):
        if time.time() > deadline - 8:
            break
        budget = min(3.0, max(0.5, deadline - time.time() - 8))
        res = _search_rank(8, time_budget=budget, seed=seed)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    return _schoolbook()
# EVOLVE-BLOCK-END
