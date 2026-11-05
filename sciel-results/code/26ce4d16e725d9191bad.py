import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _build_tensor():
    n, m, p = DIMENSIONS
    N, M, P = n*m, m*p, p*n
    T = np.zeros((N, M, P), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T, (N, M, P)

def _snap_verify(U, V, W, T, half):
    step = 0.5 if half else 1.0
    Us = np.round(U/step)*step
    Vs = np.round(V/step)*step
    Ws = np.round(W/step)*step
    recon = np.einsum('ar,br,cr->abc', Us, Vs, Ws)
    if np.allclose(recon, T, atol=1e-9):
        return Us.tolist(), Vs.tolist(), Ws.tolist()
    return None

def _als(T, r, max_iter, rng, seed_mode):
    N, M, P = T.shape
    if seed_mode == 0:
        U = rng.standard_normal((N, r)); V = rng.standard_normal((M, r)); W = rng.standard_normal((P, r))
    else:
        U, _ = np.linalg.svd(T.reshape(N, M*P), full_matrices=False)
        V, _ = np.linalg.svd(T.transpose(1,0,2).reshape(M, N*P), full_matrices=False)
        W, _ = np.linalg.svd(T.transpose(2,0,1).reshape(P, N*M), full_matrices=False)
        U = U[:, :r] + 0.05*rng.standard_normal((N, r))
        V = V[:, :r] + 0.05*rng.standard_normal((M, r))
        W = W[:, :r] + 0.05*rng.standard_normal((P, r))
    for _ in range(max_iter):
        MW = np.einsum('br,cr->bcr', V, W)
        for a in range(N):
            sol, *_ = np.linalg.lstsq(MW.reshape(M*P, r), T[a].reshape(M*P), rcond=None)
            U[a] = sol
        UW = np.einsum('ar,cr->acr', U, W)
        for b in range(M):
            sol, *_ = np.linalg.lstsq(UW.reshape(N*P, r), T[:, b, :].reshape(N*P), rcond=None)
            V[b] = sol
        UV = np.einsum('ar,br->abr', U, V)
        for c in range(P):
            sol, *_ = np.linalg.lstsq(UV.reshape(N*M, r), T[:, :, c].reshape(N*M), rcond=None)
            W[c] = sol
    return U, V, W

def _schoolbook():
    n, m, p = DIMENSIONS
    N, M, P = n*m, m*p, p*n
    rank = n*m*p
    U = [[0.0]*rank for _ in range(N)]
    V = [[0.0]*rank for _ in range(M)]
    W = [[0.0]*rank for _ in range(P)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 110.0
    T, shape = _build_tensor()
    N, M, P = shape
    fallback = _schoolbook()
    candidate_ranks = [59, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30]
    best = None
    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
        for seed_mode in (1, 0):
            if time.time() > deadline - 8:
                break
            rng = np.random.default_rng(100000 + r*31 + seed_mode)
            U, V, W = _als(T, r, max_iter=35, rng=rng, seed_mode=seed_mode)
            for half in (False, True):
                res = _snap_verify(U, V, W, T, half=half)
                if res is not None:
                    U2, V2, W2 = res
                    if len(U2)==N and len(V2)==M and len(W2)==P and len(U2[0])==r:
                        best = {"U": U2, "V": V2, "W": W2}
                        break
            if best is not None:
                break
        if best is not None:
            break
    if best is not None:
        return best
    return fallback
