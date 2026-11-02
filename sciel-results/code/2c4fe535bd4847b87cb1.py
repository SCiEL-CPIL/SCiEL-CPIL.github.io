import numpy as np
import time

def _strassen_block(U, V, W, offset, block):
    n, m, p = 2, 2, 4
    terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    for t in range(7):
        r = offset + t
        uc, vc, wc = terms[t]
        for idx, val in uc:
            U[idx, r] += val
        for idx, val in vc:
            j, kl = divmod(idx, 2)
            V[j * p + kl + block * 2, r] += val
        for idx, val in wc:
            kl, i = divmod(idx, 2)
            W[(kl + block * 2) * n + i, r] += val

def _fallback():
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    _strassen_block(U, V, W, 0, 0)
    _strassen_block(U, V, W, 7, 1)
    return U, V, W

def _check(U, V, W):
    for i in range(2):
        for j in range(2):
            for k in range(4):
                s = np.dot(U[i*2+j], V[j*4+k] * W[k*2+i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _am_rank13(budget):
    n, m, p, R = 2, 2, 4, 13
    t0 = time.time()
    for trial in range(50):
        if time.time() - t0 > budget:
            break
        np.random.seed(1000 + trial)
        U = np.random.randn(4, R) * 0.3
        V = np.random.randn(8, R) * 0.3
        W = np.random.randn(8, R) * 0.3
        best_e = 1e18
        for it in range(800):
            if time.time() - t0 > budget:
                break
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j*4+k] * W[k*2+i] for k in range(p)])
                    U[i*2+j] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i*2+j] * W[k*2+i] for i in range(n)])
                    V[j*4+k] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
            for i in range(n):
                for k in range(p):
                    A = np.array([U[i*2+j] * V[j*4+k] for j in range(m)])
                    W[k*2+i] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
            e = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        d = np.dot(U[i*2+j], V[j*4+k] * W[k*2+i]) - 1.0
                        e += d * d
            e = e ** 0.5
            if e < best_e:
                best_e = e
                bU, bV, bW = U.copy(), V.copy(), W.copy()
            if e < 1e-12:
                return bU, bV, bW
        if best_e < 1e-10:
            return bU, bV, bW
    return None

def construct():
    t0 = time.time()
    dl = t0 + 138.0
    res = _am_rank13(120.0)
    if res is not None and _check(*res):
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if time.time() > dl:
        U, V, W = _fallback()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    U, V, W = _fallback()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
