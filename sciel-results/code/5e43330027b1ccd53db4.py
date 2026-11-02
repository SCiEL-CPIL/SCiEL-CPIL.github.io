"""Structural rewrite: active randomized rank search (secant/alternating) with 42-rank fallback."""
import time
import numpy as np

n, m, p = 2, 4, 6
_START = time.time()
_DEADLINE = _START + 108.0
T = None


def _elapsed():
    return time.time() - _START


def _build_tensor():
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _res(U, V, W):
    return np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)


def _check(U, V, W, tol=1e-7):
    if _elapsed() > _DEADLINE:
        return False
    return _res(U, V, W) < tol


def _strassen_2x2():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1;U[3,0]=1;V[0,0]=1;V[3,0]=1;W[0,0]=1;W[3,0]=1
    U[2,1]=1;U[3,1]=1;V[0,1]=1;W[2,1]=1
    U[0,2]=1;V[1,2]=1;V[3,2]=-1;W[1,2]=1
    U[3,3]=1;V[2,3]=1;V[0,3]=-1;W[0,3]=-1;W[2,3]=1
    U[0,4]=1;U[1,4]=1;V[3,4]=1;W[0,4]=1;W[1,4]=1
    U[2,5]=1;U[0,5]=-1;V[0,5]=1;V[1,5]=1;W[0,5]=1
    U[1,6]=1;U[3,6]=-1;V[2,6]=1;V[3,6]=1;W[3,6]=1
    return U, V, W


def _base_42():
    Us, Vs, Ws = _strassen_2x2()
    R = 42
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    U0=[0,1,4,5]; U1=[2,3,6,7]
    VB=[[0,1,6,7],[2,3,8,9],[4,5,10,11],
        [12,13,18,19],[14,15,20,21],[16,17,22,23]]
    WC=[[0,2,1,3],[4,6,5,7],[8,10,9,11]]
    cfg=[(U0,VB[0],WC[0],0),(U1,VB[3],WC[0],7),
         (U0,VB[1],WC[1],14),(U1,VB[4],WC[1],21),
         (U0,VB[2],WC[2],28),(U1,VB[5],WC[2],35)]
    for ur,vr,wr,off in cfg:
        for r in range(7):
            for idx in range(4):
                U[ur[idx],off+r]=Us[idx,r]
                V[vr[idx],off+r]=Vs[idx,r]
                W[wr[idx],off+r]=Ws[idx,r]
    return U, V, W


def _alternating(U, V, W, iters=400, tol=1e-7):
    """Alternating least squares to drive residual to zero for fixed R."""
    for it in range(iters):
        if _elapsed() > _DEADLINE:
            break
        # Update U: for each a, U[a,:] solves V^T (W*slice_a)^T ... 
        # Residual R_abc = T_abc - sum_r U_ar V_br W_cr
        # Fix V,W: for each a, minimize over U_a: sum_b,c (T_abc - U_a·(V_b⊗W_c))²
        # = linear in U_a. Build A_a (R x (m*p)) with A_a[r, b,c] = V_br W_cr, target t_a[b,c]=T[a,b,c]
        # Solve U_a = (A_a^T A_a)^-1 A_a^T t_a  (least squares)
        # But A_a depends on V,W which are fixed in this step.
        VW = np.einsum('br,cr->bcr', V, W)  # (m*p, n*p, R)
        for a in range(n*m):
            Aa = VW[:, :, :].transpose(2, 0, 1).reshape(R, -1)  # (R, m*p*n*p)
            ta = T[a].reshape(-1)
            U[a] = np.linalg.lstsq(Aa, ta, rcond=None)[0]
        # Update V
        UW = np.einsum('ar,cr->acr', U, W)  # (n*m, n*p, R)
        for b in range(m*p):
            Ab = UW[:, :, :].transpose(2, 0, 1).reshape(R, -1)
            tb = T[:, b, :].reshape(-1)
            V[b] = np.linalg.lstsq(Ab, tb, rcond=None)[0]
        # Update W
        UV = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
        for c in range(n*p):
            Ac = UV[:, :, :].transpose(2, 0, 1).reshape(R, -1)
            tc = T[:, :, c].reshape(-1)
            W[c] = np.linalg.lstsq(Ac, tc, rcond=None)[0]
        if _res(U, V, W) < tol:
            break
    return U, V, W


def construct():
    global T
    T = _build_tensor()
    U42, V42, W42 = _base_42()
    best_rank = 42
    best = (U42.copy(), V42.copy(), W42.copy())
    if not _check(U42, V42, W42):
        # fallback schoolbook
        R0 = n*m*p
        Uf=np.zeros((n*m,R0)); Vf=np.zeros((m*p,R0)); Wf=np.zeros((p*n,R0))
        c=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j,c]=1; Vf[j*p+k,c]=1; Wf[k*n+i,c]=1; c+=1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # Try to find lower rank via alternating LS from random inits
    rng = np.random.default_rng(0)
    for R in range(36, 42):
        if _elapsed() > _DEADLINE - 5:
            break
        found = False
        for trial in range(30):
            if _elapsed() > _DEADLINE - 5:
                break
            U = rng.standard_normal((n*m, R))
            V = rng.standard_normal((m*p, R))
            W = rng.standard_normal((p*n, R))
            U, V, W = _alternating(U, V, W, iters=200, tol=1e-7)
            if _res(U, V, W) < 1e-7:
                best_rank = R
                best = (U.copy(), V.copy(), W.copy())
                found = True
                break
        if found:
            break
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
