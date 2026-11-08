import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 8, 24, 12
    
    def compute_error(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        temp = np.einsum('ijr,jkr->ijkr', U_r, V_r)
        result = np.einsum('ijkr,kir->ijk', temp, W_r)
        return np.max(np.abs(result - 1.0))
    
    # Schoolbook
    full_rank = n*m*p  # 48
    U_full = np.zeros((nm, full_rank))
    V_full = np.zeros((mp, full_rank))
    W_full = np.zeros((pn, full_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_full[i*m+j, col] = 1
                V_full[j*p+k, col] = 1
                W_full[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    best_rank = full_rank
    
    def als_step(U, V, W):
        """One ALS iteration."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # Update U
        for i in range(n):
            for j in range(m):
                A_sys = V_r[j, :, :] * W_r[:, i, :]  # (p, r)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        U_r = U.reshape(n, m, r)
        W_r = W.reshape(p, n, r)
        for j in range(m):
            for k in range(p):
                A_sys = U_r[:, j, :] * W_r[k, :, :]  # (n, r)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        for k in range(p):
            for i in range(n):
                A_sys = U_r[i, :, :] * V_r[:, k, :]  # (m, r)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)[0]
                except:
                    pass
    
    def try_rank(r, seed=42, max_iters=80):
        np.random.seed(seed)
        
        # Initialize from schoolbook subset
        if r <= full_rank:
            cols = np.random.choice(full_rank, size=r, replace=False)
            U = U_full[:, cols].astype(float).copy()
            V = V_full[:, cols].astype(float).copy()
            W = W_full[:, cols].astype(float).copy()
        else:
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            als_step(U, V, W)
            
            if it % 10 == 0:
                err = compute_error(U, V, W)
                if err < 1e-8:
                    return U.copy(), V.copy(), W.copy()
        
        err = compute_error(U, V, W)
        if err < 1e-8:
            return U.copy(), V.copy(), W.copy()
        return None
    
    # Try ranks
    for r in range(full_rank - 1, 15, -1):
        if time.time() > deadline - 60:
            break
        
        for seed in [42, 123, 7, 999]:
            if time.time() > deadline - 60:
                break
            result = try_rank(r, seed=seed, max_iters=60)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = r
                break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
