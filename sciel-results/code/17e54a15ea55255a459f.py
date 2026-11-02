"""Rewritten: random-init + gradient refinement to reduce matrix-mult tensor rank."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]
N, M, P = DIMENSIONS
TOTAL_TIME = 150.0
SAFETY = 18.0          # safety margin so we return well before the kill
DEADLINE = time.time() + (TOTAL_TIME - SAFETY)


def _tens(U, V, W, r):
    # sum_r U_r (n*m) x V_r (m*p) x W_r (p*n)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    return np.tensordot(U, V, axes=([1], [1])) * W[:, None, None]


def _resid(U, V, W, T):
    return _tens(U, V, W, len(U[0]) if len(U) else 0) - T


def _norm(x):
    return float(np.linalg.norm(x))


def _refine(U0, V0, W0, T, r, budget):
    """Gradient descent at fixed rank r. Returns (U,V,W) with minimal residual."""
    U = U0.astype(float).copy()
    V = V0.astype(float).copy()
    W = W0.astype(float).copy()
    T = np.asarray(T, dtype=float)
    best = _norm(_tens(U, V, W, r) - T)
    lr = 1e-3
    it = 0
    while time.time() < budget and it < 40000:
        it += 1
        Tens = _tens(U, V, W, r)
        R = Tens - T
        # grads: dR/dU = einsum('rabc,rc->rab', R, W) ; etc.
        gU = np.einsum('rabc,rc->rab', R, W)
        gV = np.einsum('rabc,ra->rbc', R, U)
        gW = np.einsum('rabc,ab->rc', R, U * V)
        # adaptive lr (simple backtracking on residual)
        for _ in range(4):
            nU = U - lr * gU
            nV = V - lr * gV
            nW = W - lr * gW
            nr = _norm(_tens(nU, nV, nW, r) - T)
            if nr < best:
                U, V, W = nU, nV, nW
                best = nr
                break
            lr *= 0.5
        if best < 1e-7:
            break
        if it % 200 == 0:
            lr = min(lr * 1.5, 5e-2)
    return U, V, W


def _validate(U, V, W, T, tol=1e-6):
    return _norm(_tens(U, V, W, len(U[0])) - T) < tol


def construct():
    n, m, p = N, M, P
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W

    U0, V0, W0 = schoolbook()
    best = (U0, V0, W0)
    best_rank = n * m * p

    rng = np.random.default_rng(0)
    # Try to reduce rank from 95 downward.
    for r in range(n * m * p - 1, 0, -1):
        if time.time() > DEADLINE:
            break
        # per-rank budget: a few seconds of attempts
        rank_budget = min(DEADLINE, time.time() + 6.0)
        found = False
        for attempt in range(6):
            if time.time() > rank_budget:
                break
            # init: take top-r columns of a perturbed schoolbook, plus noise
            U = U0[:, :r].copy() + 0.05 * rng.standard_normal((nm, r))
            V = V0[:, :r].copy() + 0.05 * rng.standard_normal((mp, r))
            W = W0[:, :r].copy() + 0.05 * rng.standard_normal((pn, r))
            # occasionally fully random
            if attempt % 3 == 2:
                U = rng.standard_normal((nm, r))
                V = rng.standard_normal((mp, r))
                W = rng.standard_normal((pn, r))
            U, V, W = _refine(U, V, W, T, r, rank_budget)
            if _validate(U, V, W, T):
                best = (U, V, W)
                best_rank = r
                found = True
                break
        if found:
            # try one more step down quickly
            continue

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
