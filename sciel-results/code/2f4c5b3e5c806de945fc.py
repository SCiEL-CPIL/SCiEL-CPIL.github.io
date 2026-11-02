import numpy as np
import time

T0 = time.time()
DEADLINE = T0 + 128.0   # safety margin under the 150 s kill

N, M, P = 4, 4, 5
def u_idx(i, j): return i * M + j
def v_idx(j, k): return j * P + k
def w_idx(k, i): return k * N + i

def _bini_4x4x4():
    Uu, Vu, Wu = {}, {}, {}
    r = 0
    for i in range(4):
        for j in range(4):
            Uu[(i, j)] = 1; Vu[(j, 0)] = 1; Wu[(0, i)] = 1; r += 1
    for i in range(4):
        for j in range(1, 4):
            for k in range(1, 4):
                Uu[(i, j)] = 1; Vu[(j, k)] = 1; Wu[(k, i)] = 1; r += 1
    for i in range(1, 4):
        for j in range(1, 4):
            Uu[(i, j)] = 1; Vu[(j, 0)] = 1; Wu[(0, i)] = -1; r += 1
    assert r == 49
    return Uu, Vu, Wu

def _check(U, V, W):
    Uv = U.reshape(N, M, -1)
    Vv = V.reshape(M, P, -1)
    Wv = W.reshape(P, N, -1)
    T = np.einsum('ijr,jkr,kir->kji', Uv, Vv, Wv)
    return np.allclose(T, 1.0, atol=1e-6)

def _build_base():
    U = np.zeros((N * M, 65)); V = np.zeros((M * P, 65)); W = np.zeros((P * N, 65))
    Uu, Vu, Wu = _bini_4x4x4()
    r = 0
    for (i, j), a in Uu.items(): U[u_idx(i, j), r] = a
    for (j, k), b in Vu.items(): V[v_idx(j, k), r] = b
    for (k, i), c in Wu.items(): W[w_idx(k, i), r] = c
    r += 1
    for i in range(N):
        for j in range(M):
            U[u_idx(i, j), r] = 1
            V[v_idx(j, 4), r] = 1
            W[w_idx(4, i), r] = 1
            r += 1
    assert r == 65
    return U, V, W

def _trim(U, V, W):
    keep = [r for r in range(U.shape[1])
            if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0)]
    return U[:, keep], V[:, keep], W[:, keep], len(keep)

def _reduce_one(U, V, W, rng):
    """Re-solve one factor of one term (least squares) while holding the other
    two factors fixed; returns (U,V,W) if the full decomposition still holds."""
    R = U.shape[1]
    r = int(rng.integers(0, R))
    which = int(rng.integers(0, 3))
    Uf = U.copy(); Vf = V.copy(); Wf = W.copy()
    # T_other: contribution of all terms except r
    Tout = np.zeros((P, M, N))
    Uv = Uf.reshape(N, M, R); Vv = Vf.reshape(M, P, R); Wv = Wf.reshape(P, N, R)
    for rp in range(R):
        if rp == r:
            continue
        Tout += np.einsum('ij,jk,ki->kji', Uv[:, :, rp], Vv[:, :, rp], Wv[:, :, rp])
    if which == 2:   # solve W[k,i] for term r: coeff[k,i]=sum_j U[i,j,r]*V[j,k,r]
        coeff = np.zeros((P, N))
        for k in range(P):
            for j in range(M):
                vv = Vv[j, k, r]
                if vv == 0: continue
                for i in range(N):
                    coeff[k, i] += Uv[i, j, r] * vv
        A = coeff.ravel(); b = (1.0 - Tout).ravel()
        denom = float(np.dot(A, A))
        x = float(np.dot(A, b)) / denom if denom > 1e-12 else 0.0
        Wf[:, r] = 0.0
        for k in range(P):
            for i in range(N):
                Wf[w_idx(k, i), r] = x
    elif which == 0:  # solve U[i,j] for term r: coeff[i,j]=sum_k V[j,k,r]*W[k,i,r]
        coeff = np.zeros((N, M))
        for k in range(P):
            for j in range(M):
                vv = Vv[j, k, r]
                if vv == 0: continue
                for i in range(N):
                    coeff[i, j] += vv * Wv[k, i, r]
        A = coeff.ravel(); b = (1.0 - Tout).ravel()
        denom = float(np.dot(A, A))
        x = float(np.dot(A, b)) / denom if denom > 1e-12 else 0.0
        Uf[:, r] = 0.0
        for i in range(N):
            for j in range(M):
                Uf[u_idx(i, j), r] = x
    else:             # solve V[j,k] for term r: coeff[j,k]=sum_i U[i,j,r]*W[k,i,r]
        coeff = np.zeros((M, P))
        for i in range(N):
            for j in range(M):
                uu = Uv[i, j, r]
                if uu == 0: continue
                for k in range(P):
                    coeff[j, k] += uu * Wv[k, i, r]
        A = coeff.ravel(); b = (1.0 - Tout).ravel()
        denom = float(np.dot(A, A))
        x = float(np.dot(A, b)) / denom if denom > 1e-12 else 0.0
        Vf[:, r] = 0.0
        for j in range(M):
            for k in range(P):
                Vf[v_idx(j, k), r] = x
    if _check(Uf, Vf, Wf):
        return Uf, Vf, Wf
    return None

def construct():
    U, V, W = _build_base()
    if not _check(U, V, W):
        U = np.zeros((N * M, N * M * P)); V = np.zeros((M * P, N * M * P)); W = np.zeros((P * N, N * M * P))
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[u_idx(i, j), c] = 1; V[v_idx(j, k), c] = 1; W[w_idx(k, i), c] = 1; c += 1
        return {"U": U, "V": V, "W": W}
    best = (U, V, W, 65)
    rng = np.random.default_rng(7)
    while time.time() < DEADLINE - 8 and best[3] > 64:
        U, V, W, cur = best
        for _ in range(300):
            if time.time() > DEADLINE - 8:
                break
            res = _reduce_one(U, V, W, rng)
            if res is None:
                continue
            U2, V2, W2, rk = _trim(*res)
            if rk < cur:
                best = (U2, V2, W2, rk)
                break
    U, V, W, _ = best
    return {"U": U, "V": V, "W": W}
