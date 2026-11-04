import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def verify(U, V, W, tol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    def als(rank, seed, max_iter=300, tol=1e-12):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            VW = V @ W.T
            U = np.linalg.lstsq(VW.T, T.transpose(1,2,0), rcond=None)[0].T
            
            UW = U @ W.T
            V = np.linalg.lstsq(UW.T, T.transpose(0,2,1), rcond=None)[0].T
            
            A_mat = (U[:, :, None] * V[:, None, :]).reshape(nm*mp, rank)
            T_flat = T.reshape(nm*mp, pn)
            W = np.linalg.lstsq(A_mat.T, T_flat, rcond=None)[0].T
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(recon - T)) < tol:
                break
        
        if verify(U, V, W, tol=1e-8):
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if verify(U_r, V_r, W_r, tol=1e-6):
                return [U_r.tolist(), V_r.tolist(), W_r.tolist()]
            return [U.tolist(), V.tolist(), W.tolist()]
        return None
    
    best = None
    for rank in range(18, 37):
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            result = als(rank, seed=seed*7919 + rank*31)
            if result is not None:
                best = result
                break
        if best is not None:
            U, V, W = best
            return {"U": U, "V": V, "W": W}
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
