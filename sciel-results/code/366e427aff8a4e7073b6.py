import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check(U, V, W, T):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    result = np.zeros(T.shape, dtype=np.float64)
    for r in range(Ua.shape[1]):
        result += np.einsum('u,v,w->uvw', Ua[:,r], Va[:,r], Wa[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-8)

def _build_rank18(n, m, p):
    """Strassen on (col0,col1) + Strassen on (col2,col3) + schoolbook col4."""
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def strassen_block(c0, c1, base):
        # r=base+0: (A00+A11)(B0c0+B1c1) -> contributes to C0c0, C1c1
        U[0][base] = 1; U[3][base] = 1
        V[c0][base] = 1; V[5+c1][base] = 1
        W[c0][base] = 1; W[5+c1][base] = 1
        # r=base+1: (A10+A11)B0c0 -> C1c0, C1c1
        U[2][base+1] = 1; U[3][base+1] = 1
        V[c0][base+1] = 1
        W[5+c0][base+1] = 1; W[5+c1][base+1] = -1
        # r=base+2: A00(B1c1-B0c1) -> C0c1, C1c1
        U[0][base+2] = 1
        V[5+c1][base+2] = 1; V[c1][base+2] = -1
        W[c1][base+2] = 1; W[5+c1][base+2] = 1
        # r=base+3: A11(B0c0-B1c0) -> C0c0, C1c0
        U[3][base+3] = 1
        V[c0][base+3] = 1; V[5+c0][base+3] = -1
        W[c0][base+3] = 1; W[5+c0][base+3] = 1
        # r=base+4: (A00+A01)B1c1 -> C0c0, C0c1
        U[0][base+4] = 1; U[1][base+4] = 1
        V[5+c1][base+4] = 1
        W[c0][base+4] = -1; W[c1][base+4] = 1
        # r=base+5: (A10-A00)(B0c0+B0c1) -> C1c1
        U[2][base+5] = 1; U[0][base+5] = -1
        V[c0][base+5] = 1; V[c1][base+5] = 1
        W[5+c1][base+5] = 1
        # r=base+6: (A01-A11)(B1c0+B1c1) -> C0c0
        U[1][base+6] = 1; U[3][base+6] = -1
        V[5+c0][base+6] = 1; V[5+c1][base+6] = 1
        W[c0][base+6] = 1
    
    strassen_block(0, 1, 0)
    strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    U[0][14] = 1; V[4][14] = 1; W[4][14] = 1
    U[1][15] = 1; V[9][15] = 1; W[4][15] = 1
    U[2][16] = 1; V[4][16] = 1; W[9][16] = 1
    U[3][17] = 1; V[9][17] = 1; W[9][17] = 1
    
    return U, V, W

def _als_solve(T, U, V, W, R, which):
    """Solve one factor given the other two."""
    Ta = np.array(T, dtype=np.float64)
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    if which == 'U':
        # For each u: T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # Design matrix row (v,w): [V[v,0]*W[w,0], ..., V[v,R-1]*W[w,R-1]]
        VW = np.einsum('vr,wr->vwr', Va, Wa)  # (10, R, 10)
        VWf = VW.transpose(0, 2, 1).reshape(-1, R)  # (100, R)
        U_new = np.zeros((4, R))
        for u in range(4):
            b = Ta[u].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(VWf, b, rcond=None)
            U_new[u] = sol
        return U_new
    elif which == 'V':
        # For each v: T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        UW = np.einsum('ur,wr->uwr', Ua, Wa)  # (4, 10, R)
        UWf = UW.transpose(0, 2, 1).reshape(-1, R)  # (40, R)
        V_new = np.zeros((10, R))
        for v in range(10):
            b = Ta[:, v, :].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(UWf, b, rcond=None)
            V_new[v] = sol
        return V_new
    else:  # 'W'
        UV = np.einsum('ur,vr->uvr', Ua, Va)  # (4, 10, R)
        UVf = UV.reshape(-1, R)  # (40, R)
        W_new = np.zeros((10, R))
        for w in range(10):
            b = Ta[:, :, w].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(UVf, b, rcond=None)
            W_new[w] = sol
        return W_new

def _als_iterate(T, U, V, W, R, max_iters=100, tol=1e-10):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    prev_err = np.inf
    for it in range(max_iters):
        Ua = _als_solve(T, Ua, Va, Wa, R, 'U')
        Va = _als_solve(T, Ua, Va, Wa, R, 'V')
        Wa = _als_solve(T, Ua, Va, Wa, R, 'W')
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', Ua[:,r], Va[:,r], Wa[:,r]) - T)**2)
        err = np.sqrt(err)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return Ua, Va, Wa

def _try_rank17(T, deadline):
    """Try to find rank-17 by modifying rank-18 structure."""
    n, m, p = 2, 2, 5
    U18, V18, W18 = _build_rank18(n, m, p)
    U18a = np.array(U18, dtype=np.float64)
    V18a = np.array(V18, dtype=np.float64)
    W18a = np.array(W18, dtype=np.float64)
    
    # Try removing each schoolbook term and running ALS
    for rem in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        idx = [i for i in range(18) if i != rem]
        U0 = U18a[:, idx].copy()
        V0 = V18a[:, idx].copy()
        W0 = W18a[:, idx].copy()
        np.random.seed(rem * 100 + 7)
        U0 += np.random.randn(4, 17) * 0.01
        V0 += np.random.randn(10, 17) * 0.01
        W0 += np.random.randn(10, 17) * 0.01
        Uf, Vf, Wf = _als_iterate(T, U0, V0, W0, 17, max_iters=80)
        if _check(Uf.tolist(), Vf.tolist(), Wf.tolist(), T):
            return Uf.tolist(), Vf.tolist(), Wf.tolist()
    
    # Try merging a schoolbook term into a Strassen term
    for st in range(14):
        if time.time() > deadline:
            break
        for sc in [14, 15, 16, 17]:
            if time.time() > deadline:
                break
            U0 = U18a.copy()
            V0 = V18a.copy()
            W0 = W18a.copy()
            U0[:, st] += U18a[:, sc]
            V0[:, st] += V18a[:, sc]
            W0[:, st] += W18a[:, sc]
            idx = [i for i in range(18) if i != sc]
            U0 = U0[:, idx]
            V0 = V0[:, idx]
            W0 = W0[:, idx]
            np.random.seed(st * 1000 + sc)
            U0 += np.random.randn(4, 17) * 0.005
            V0 += np.random.randn(10, 17) * 0.005
            W0 += np.random.randn(10, 17) * 0.005
            Uf, Vf, Wf = _als_iterate(T, U0, V0, W0, 17, max_iters=60)
            if _check(Uf.tolist(), Vf.tolist(), Wf.tolist(), T):
                return Uf.tolist(), Vf.tolist(), Wf.tolist()
    
    return None

def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Try rank 17
    result = _try_rank17(T, deadline)
    if result is not None:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback to rank 18
    U, V, W = _build_rank18(n, m, p)
    return {"U": U, "V": V, "W": W}
