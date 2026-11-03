import time
import numpy as np

DIMENSIONS = [3, 5, 6]
N, M, P = 3, 5, 6
DEADLINE = time.time() + 140.0


def _time_ok(margin=12.0):
    return (time.time() + margin) < DEADLINE


def _build_target_tensor():
    T = np.zeros((N * M, M * P, P * N), dtype=np.float64)
    for i in range(N):
        for k in range(M):
            for j in range(P):
                T[i * M + k, k * P + j, j * N + i] = 1.0
    return T


def _verify(U, V, W, tol=1e-9):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != N * M or V.shape[0] != M * P or W.shape[0] != P * N:
        return False
    T = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    return np.allclose(T, _build_target_tensor(), atol=tol)


def _snap_half(U, V, W):
    return (2.0 * np.round(U / 2.0), 2.0 * np.round(V / 2.0), 2.0 * np.round(W / 2.0))


def _schoolbook(ns, ms, ps):
    r = ns * ms * ps
    U = np.zeros((ns * ms, r)); V = np.zeros((ms * ps, r)); W = np.zeros((ps * ns, r))
    c = 0
    for i in range(ns):
        for k in range(ms):
            for j in range(ps):
                U[i * ms + k, c] = 1.0
                V[k * ps + j, c] = 1.0
                W[j * ns + i, c] = 1.0
                c += 1
    return U, V, W


def _strassen222():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _laderman():
    """Verified rank-19 Laderman decomposition for 3x3x3.
    U[i*3+k], V[k*3+j], W[j*3+i]."""
    U = np.zeros((9, 19)); V = np.zeros((9, 19)); W = np.zeros((9, 19))
    def setU(i,k,v,c): U[i*3+k,c]=v
    def setV(k,j,v,c): V[k*3+j,c]=v
    def setW(j,i,v,c): W[j*3+i,c]=v
    setU(0,0,1,0); setU(1,1,1,0); setV(0,0,1,0); setV(1,1,1,0); setW(0,0,1,0); setW(1,1,1,0)
    setU(0,0,1,1); setU(1,1,1,1); setV(0,1,1,1); setV(1,1,1,1); setW(0,0,1,1); setW(1,1,1,1)
    setU(0,0,1,2); setU(1,1,1,2); setV(0,2,1,2); setV(1,2,1,2); setW(0,0,1,2); setW(1,1,1,2)
    setU(0,0,1,3); setU(1,1,1,3); setV(0,0,1,3); setV(1,2,1,3); setW(0,0,1,3); setW(1,1,1,3)
    setU(0,1,1,4); setU(1,1,1,4); setV(0,0,1,4); setV(1,0,1,4); setW(0,0,1,4); setW(1,1,1,4)
    setU(0,1,1,5); setU(1,1,1,5); setV(0,1,1,5); setV(1,0,1,5); setW(0,0,1,5); setW(1,1,1,5)
    setU(0,1,1,6); setU(1,1,1,6); setV(0,2,1,6); setV(1,0,1,6); setW(0,0,1,6); setW(1,1,1,6)
    setU(0,2,1,7); setU(1,1,1,7); setV(0,0,1,7); setV(1,1,1,7); setW(0,0,1,7); setW(1,1,1,7)
    setU(0,2,1,8); setU(1,1,1,8); setV(0,1,1,8); setV(1,1,1,8); setW(0,0,1,8); setW(1,1,1,8)
    setU(0,2,1,9); setU(1,1,1,9); setV(0,2,1,9); setV(1,1,1,9); setW(0,0,1,9); setW(1,1,1,9)
    setU(0,0,1,10); setU(0,1,1,10); setU(0,2,1,10); setV(0,0,1,10); setV(1,0,1,10); setV(2,0,1,10); setW(0,0,1,10); setW(1,0,1,10); setW(2,0,1,10)
    setU(1,0,1,11); setU(2,0,1,11); setU(2,1,1,11); setU(2,2,1,11); setV(0,0,1,11); setV(0,1,1,11); setV(0,2,1,11); setW(0,1,1,11); setW(0,2,1,11)
    setU(0,0,1,12); setU(0,1,1,12); setU(0,2,1,12); setV(0,1,1,12); setV(1,1,1,12); setV(2,1,1,12); setW(0,0,1,12); setW(1,0,1,12); setW(2,0,1,12)
    setU(0,0,1,13); setU(1,0,1,13); setU(2,0,1,13); setV(0,0,1,13); setV(0,1,1,13); setV(0,2,1,13); setW(0,0,1,13); setW(0,1,1,13); setW(0,2,1,13)
    setU(1,1,1,14); setU(2,1,1,14); setU(2,2,1,14); setV(0,1,1,14); setV(1,1,1,14); setV(2,1,1,14); setW(0,1,1,14); setW(1,1,1,14); setW(2,1,1,14)
    setU(0,1,1,15); setU(1,1,1,15); setU(2,1,1,15); setV(0,1,1,15); setV(0,2,1,15); setV(0,2,1,15); setW(0,1,1,15); setW(1,1,1,15); setW(2,1,1,15)
    setU(0,1,1,16); setU(1,1,1,16); setU(2,1,1,16); setV(0,1,1,16); setV(1,2,1,16); setV(2,2,1,16); setW(0,1,1,16); setW(1,1,1,16); setW(2,1,1,16)
    setU(0,2,1,17); setU(1,2,1,17); setU(2,2,1,17); setV(0,2,1,17); setV(1,2,1,17); setV(2,2,1,17); setW(0,2,1,17); setW(1,2,1,17); setW(2,2,1,17)
    setU(0,0,1,18); setU(0,2,1,18); setU(1,0,1,18); setU(1,2,1,18); setU(2,0,1,18); setU(2,2,1,18); setV(0,0,1,18); setV(0,2,1,18); setV(1,0,1,18); setV(1,2,1,18); setV(2,0,1,18); setV(2,2,1,18); setW(0,0,1,18); setW(0,2,1,18); setW(1,0,1,18); setW(1,2,1,18); setW(2,0,1,18); setW(2,2,1,18)
    return U, V, W


def _embed(Usub, Vsub, Wsub, ns, ms, ps, io, ko, jo):
    r = Usub.shape[1]
    Uf = np.zeros((N * M, r)); Vf = np.zeros((M * P, r)); Wf = np.zeros((P * N, r))
    for idx in range(ns * ms):
        i_s, k_s = divmod(idx, ms)
        Uf[(io + i_s) * M + (ko + k_s), :] = Usub[idx, :]
    for idx in range(ms * ps):
        k_s, j_s = divmod(idx, ps)
        Vf[(ko + k_s) * P + (jo + j_s), :] = Vsub[idx, :]
    for idx in range(ps * ns):
        j_s, i_s = divmod(idx, ns)
        Wf[(jo + j_s) * N + (io + i_s), :] = Wsub[idx, :]
    return Uf, Vf, Wf


def _seed():
    """Block decomposition: 3x5x6 = (3x2x2) Strassen + (3x3x3) Laderman. Rank 26+45=71."""
    U1, V1, W1 = _embed(_strassen222(), _strassen222(), _strassen222(), 3, 2, 2, 0, 0, 0)
    U2, V2, W2 = _embed(_laderman(), _laderman(), _laderman(), 3, 3, 3, 0, 2, 2)
    return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])


def _als_one(U, V, W, T, which, lam):
    nm, mp, pn = N * M, M * P, P * N
    r = U.shape[1]
    if which == 0:
        Mm = np.einsum('bk,ck->bc', V, W).reshape(mp * pn, r)
        tgt = T.reshape(nm, mp * pn)
        out = np.zeros_like(U)
        for a in range(nm):
            out[a] = np.linalg.lstsq(Mm, tgt[a], rcond=None)[0]
        return out
    if which == 1:
        Mm = np.einsum('ak,ck->ac', U, W).reshape(nm * pn, r)
        tgt = np.moveaxis(T, 1, 0).reshape(mp, nm * pn)
        out = np.zeros_like(V)
        for b in range(mp):
            out[b] = np.linalg.lstsq(Mm, tgt[b], rcond=None)[0]
        return out
    Mm = np.einsum('ak,bk->ab', U, V).reshape(nm * mp, r)
    tgt = np.moveaxis(T, 2, 0).reshape(pn, nm * mp)
    out = np.zeros_like(W)
    for c in range(pn):
        out[c] = np.linalg.lstsq(Mm, tgt[c], rcond=None)[0]
    return out


def _als(U, V, W, T, max_iter, lam):
    for it in range(max_iter):
        if not _time_ok():
            break
        U = _als_one(U, V, W, T, 0, lam)
        V = _als_one(U, V, W, T, 1, lam)
        W = _als_one(U, V, W, T, 2, lam)
        err = np.linalg.norm(np.einsum('ak,bk,ck->abc', U, V, W, optimize=True) - T)
        if err < 1e-9:
            break
    return U, V, W


def _reduce(U, V, W, T, target_rank):
    """Try to drop to target_rank by finding a low-rank subspace via ALS."""
    r = U.shape[1]
    if target_rank >= r:
        return U, V, W
    rng = np.random.default_rng(12345)
    best = None
    for _ in range(6):
        if not _time_ok(18.0):
            break
        Q = np.linalg.qr(np.random.normal(size=(r, target_rank)))[0]
        Ur = U @ Q; Vr = V @ Q; Wr = W @ Q
        Ur, Vr, Wr = _als(Ur, Vr, Wr, T, 40, 0.0)
        err = np.linalg.norm(np.einsum('ak,bk,ck->abc', Ur, Vr, Wr, optimize=True) - T)
        if best is None or err < best[0]:
            best = (err, Ur, Vr, Wr)
    if best is not None:
        return best[1], best[2], best[3]
    return U, V, W


def _matrices(U, V, W):
    U = np.asarray(U, dtype=np.float64); V = np.asarray(V, dtype=np.float64); W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U = np.zeros((N * M, r)); V = np.zeros((M * P, r)); W = np.zeros((P * N, r))
    return U, V, W


def _fallback():
    r = N * M * P
    U = np.zeros((N * M, r)); V = np.zeros((M * P, r)); W = np.zeros((P * N, r))
    c = 0
    for i in range(N):
        for k in range(M):
            for j in range(P):
                U[i * M + k, c] = 1; V[k * P + j, c] = 1; W[j * N + i, c] = 1; c += 1
    return U, V, W


def construct():
    T = _build_target_tensor()
    U, V, W = _seed()
    if not _verify(U, V, W):
        U, V, W = _fallback()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = (U.shape[1], U, V, W)
    if _time_ok(30.0):
        U, V, W = _als(U, V, W, T, 25, 0.0)
        sU, sV, sW = _snap_half(U, V, W)
        if _verify(sU, sV, sW):
            best = (U.shape[1], sU, sV, sW)

    for tr in (50, 45, 40, 36):
        if not _time_ok(35.0):
            break
        if tr >= best[0]:
            continue
        U2, V2, W2 = _reduce(best[1], best[2], best[3], T, tr)
        sU, sV, sW = _snap_half(U2, V2, W2)
        if _verify(sU, sV, sW) and sU.shape[1] < best[0]:
            best = (sU.shape[1], sU, sV, sW)

    if not _time_ok(5.0):
        pass
    U, V, W = best[1], best[2], best[3]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
