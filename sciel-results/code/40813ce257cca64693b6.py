import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 4
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def _verify(U, V, W):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.abs(recon - T).max() < 1e-6
    
    def _try_round(U, V, W, grid_denoms=(1, 2, 4)):
        for d in grid_denoms:
            U_r = np.round(U * d) / d
            V_r = np.round(V * d) / d
            W_r = np.round(W * d) / d
            if _verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None
    
    def _try_rank_als(r, max_iters=300, tol=1e-8, seed=42):
        if time.time() > deadline - 3:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        mp = m * p
        pn = p * n
        nm = n * m
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Update U
            for a in range(nm):
                A_mat = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, r)
                rhs = T[a, :, :].ravel()
                U[a, :] = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
            
            # Update V
            for b in range(mp):
                A_mat = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, r)
                rhs = T[:, b, :].reshape(-1)
                V[b, :] = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
            
            # Update W
            for c in range(pn):
                A_mat = np.einsum('ar,br->abr', U, V).reshape(nm * mp, r)
                rhs = T[:, :, c].reshape(-1)
                W[c, :] = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
            
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.abs(recon - T).max()
            if err < tol:
                break
        
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.abs(recon - T).max()
        if err > 1e-4:
            return None
        
        rounded = _try_round(U, V, W)
        if rounded is not None:
            return rounded
        
        return None
    
    # Try ALS for various ranks
    best_result = None
    for r in range(20, 37):
        if time.time() > deadline - 5:
            break
        for seed in range(5):
            if time.time() > deadline - 3:
                break
            result = _try_rank_als(r, max_iters=200, seed=seed)
            if result is not None:
                best_result = (r, result)
                break
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (r, (U, V, W))
    
    r, (U, V, W) = best_result
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
