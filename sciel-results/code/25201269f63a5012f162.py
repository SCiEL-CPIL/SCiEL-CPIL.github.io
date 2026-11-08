import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W

    def tensor(U, V, W):
        return np.einsum('as,bs,cs->abc', U, V, W)

    def merr(U, V, W):
        return np.max(np.abs(tensor(U, V, W) - T))

    def rh(x):
        return np.round(2 * x) / 2

    def ok(U, V, W):
        return np.array_equal(tensor(U, V, W), T)

    def als(U, V, W, mi, t0, tl):
        r = U.shape[1]
        for _ in range(mi):
            if time.time() - t0 > tl:
                break
            Un = np.zeros_like(U)
            for a in range(nm):
                M = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, r)
                Un[a] = np.linalg.lstsq(M, T[a].ravel(), rcond=None)[0]
            Vn = np.zeros_like(V)
            for b in range(mp):
                M = (U[:, :, None] * W[None, :, :]).reshape(nm * pn, r)
                Vn[b] = np.linalg.lstsq(M, T[:, b, :].ravel(), rcond=None)[0]
            Wn = np.zeros_like(W)
            for c in range(pn):
                M = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, r)
                Wn[c] = np.linalg.lstsq(M, T[:, :, c].ravel(), rcond=None)[0]
            ch = (np.max(np.abs(Un - U)) + np.max(np.abs(Vn - V)) + np.max(np.abs(Wn - W)))
            U, V, W = Un, Vn, Wn
            if ch < 1e-14:
                break
        return U, V, W

    t0 = time.time()
    tl = 100
    U0, V0, W0 = schoolbook()

    for r in [28, 30, 32, 34]:
        if time.time() - t0 > tl:
            break
        for att in range(3):
            if time.time() - t0 > tl:
                break
            np.random.seed(r * 100 + att)
            idx = np.random.choice(36, r, replace=False)
            U, V, W = U0[:, idx].copy(), V0[:, idx].copy(), W0[:, idx].copy()
            U, V, W = als(U, V, W, 800, t0, tl)
            if merr(U, V, W) < 1e-10:
                Uh, Vh, Wh = rh(U), rh(V), rh(W)
                if ok(Uh, Vh, Wh):
                    return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}

    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
