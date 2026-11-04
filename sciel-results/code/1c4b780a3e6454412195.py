import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _build_T(n, m, p):
    NA, NB, NC = n*m, m*p, p*n
    T = np.zeros((NA, NB, NC))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _als_fit(T, rank, seed, max_iter, deadline):
    NA, NB, NC = T.shape
    rng = np.random.default_rng(seed)
    T1 = T.reshape(NA, NB*NC)
    T2 = T.transpose(1,0,2).reshape(NB, NA*NC)
    T3 = T.transpose(2,0,1).reshape(NC, NA*NB)
    
    U = rng.standard_normal((NA, rank))
    V = rng.standard_normal((NB, rank))
    W = rng.standard_normal((NC, rank))
    
    def M1(V, W):
        return (V.T[:, :, None] * W.T[:, None, :]).reshape(rank, NB*NC).T
    def M2(U, W):
        return (U.T[:, :, None] * W.T[:, None, :]).reshape(rank, NA*NC).T
    def M3(U, V):
        return (U.T[:, :, None] * V.T[:, None, :]).reshape(rank, NA*NB).T
    
    best_res = np.inf
    for it in range(max_iter):
        if time.time() > deadline:
            break
        M = M1(V, W)
        U = np.linalg.lstsq(M, T1.T, rcond=None)[0].T
        M = M2(U, W)
        V = np.linalg.lstsq(M, T2.T, rcond=None)[0].T
        M = M3(U, V)
        W = np.linalg.lstsq(M, T3.T, rcond=None)[0].T
        R = T - np.einsum('ar,br,cr->abc', U, V, W)
        res = np.linalg.norm(R)
        if res < best_res:
            best_res = res
            best = (U.copy(), V.copy(), W.copy())
        if res < 1e-10:
            break
    U, V, W = best if np.isfinite(best_res) else (U, V, W)
    return U, V, W, best_res

def _try_rank(T, rank, deadline):
    NA, NB, NC = T.shape
    max_iter = 150
    for seed in range(30):
        if time.time() > deadline - 15:
            break
        U, V, W, res = _als_fit(T, rank, seed, max_iter, deadline - 15)
        if res < 1e-6:
            # try rounding
            for scale in (2,):
                Ur = np.round(U*scale)/scale
                Vr = np.round(V*scale)/scale
                Wr = np.round(W*scale)/scale
                Rr = np.einsum('ar,br,cr->abc', Ur, Vr, Wr) - T
                if np.max(np.abs(Rr)) < 1e-9:
                    return (Ur, Vr, Wr)
            # integer rounding
            Ur = np.round(U)
            Vr = np.round(V)
            Wr = np.round(W)
            Rr = np.einsum('ar,br,cr->abc', Ur, Vr, Wr) - T
            if np.max(np.abs(Rr)) < 1e-9:
                return (Ur, Vr, Wr)
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_T(n, m, p)
    deadline = time.time() + 110
    for rank in range(27, n*m*p + 1):
        if time.time() > deadline - 20:
            break
        sol = _try_rank(T, rank, deadline)
        if sol is not None:
            Ur, Vr, Wr = sol
            return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    return _schoolbook(n, m, p)
