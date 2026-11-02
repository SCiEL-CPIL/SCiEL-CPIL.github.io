import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START = time.time()
_TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    r0 = n * m * p

    # Schoolbook decomposition
    U0 = np.zeros((nm, r0))
    V0 = np.zeros((mp, r0))
    W0 = np.zeros((pn, r0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    best_rank, best_U, best_V, best_W = r0, U0, V0, W0

    # Try to find a lower-rank decomposition
    for r in range(r0 - 1, max(r0 - 5, 30), -1):
        if time.time() - _START > _TIME_LIMIT:
            break
        res = _try_rank(n, m, p, r)
        if res is not None:
            best_rank, best_U, best_V, best_W = r, res[0], res[1], res[2]
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_rank(n, m, p, r, n_trials=3):
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]

    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT:
            return None
        rng = np.random.RandomState(t * 7919 + r)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)

        for it in range(500):
            if time.time() - _START > _TIME_LIMIT:
                return None
            Ug = np.zeros_like(U)
            Vg = np.zeros_like(V)
            Wg = np.zeros_like(W)
            loss = 0.0
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                uv = U[ui] * V[vi]
                uw = U[ui] * W[wi]
                s = np.dot(U[ui], vw)
                e = s - 1.0
                loss += e * e
                Ug[ui] += e * vw
                Vg[vi] += e * uw
                Wg[wi] += e * uv

            if loss < 1e-14:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if _verify(n, m, p, U, V, W):
                    return (U, V, W)

            lr = 0.02 / (1.0 + it * 0.002)
            U -= lr * Ug
            V -= lr * Vg
            W -= lr * Wg

    return None

def _verify(n, m, p, U, V, W):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True
