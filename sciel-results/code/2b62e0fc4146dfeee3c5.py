import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
_START = time.time()
_DEADLINE = 120.0


def _basis():
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    return terms


def _verify(U, V, W, n, m, p):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _try_solve(basis, r, n, m, p, rng):
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for c in range(r):
        t = basis[c]
        U[t[0] * m + t[1], c] = 1.0
        V[t[1] * p + t[2], c] = 1.0
        W[t[2] * n + t[0], c] = 1.0
    for c in range(r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) == basis[c]:
                        continue
                    a = U[i * m + j, c]
                    b = V[j * p + k, c]
                    d = W[k * n + i, c]
                    if a == 0 or b == 0 or d == 0:
                        continue
                    x = rng.random()
                    y = rng.random()
                    z = rng.random()
                    U[i * m + j, c] = a + x
                    V[j * p + k, c] = b + y
                    W[k * n + i, c] = d + z
    A = np.zeros((n * m * p, r))
    for c in range(r):
        for i in range(n):
            ui = U[i * m:i * m + m, c]
            for k in range(p):
                wk = W[k * n:(k + 1) * n, c]
                for j in range(m):
                    A[(i * m + j) * p + k, c] = ui[j] * V[j * p + k, c] * wk[i]
    rhs = np.ones(n * m * p)
    sol, res, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
    if rank < r:
        return None
    if np.linalg.norm(A @ sol - rhs) > 1e-5:
        return None
    U2 = np.zeros((n * m, r))
    V2 = np.zeros((m * p, r))
    W2 = np.zeros((p * n, r))
    for c in range(r):
        U2[:, c] = U[:, c]
        V2[:, c] = V[:, c]
        W2[:, c] = W[:, c]
    for c in range(r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) == basis[c]:
                        continue
                    a = U[i * m + j, c]
                    b = V[j * p + k, c]
                    d = W[k * n + i, c]
                    if a == 0 or b == 0 or d == 0:
                        continue
                    U2[i * m + j, c] = a + sol[(i * m + j) * p + k]
                    V2[j * p + k, c] = b + sol[(i * m + j) * p + k]
                    W2[k * n + i, c] = d + sol[(i * m + j) * p + k]
    if not _verify(U2, V2, W2, n, m, p):
        return None
    return U2, V2, W2


def construct():
    n, m, p = DIMENSIONS
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    basis = _basis()
    L = len(basis)
    best = (rank0, U0, V0, W0)
    rng = random.Random(12345)

    def _mat(M):
        return [[float(v) for v in row] for row in M]

    for r in range(15, min(rank0, L) + 1):
        if time.time() > _DEADLINE - 15.0:
            break
        if r > best[0]:
            break
        for attempt in range(200):
            if time.time() > _DEADLINE - 15.0:
                break
            idx = list(range(L))
            rng.shuffle(idx)
            sub = [basis[i] for i in idx[:r]]
            res = _try_solve(sub, r, n, m, p, rng)
            if res is not None:
                best = (r, res[0], res[1], res[2])
                break

    r, U, V, W = best
    return {"U": _mat(U), "V": _mat(V), "W": _mat(W)}
