import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    target_rank = 17
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_error = float('inf')
    best_UVW = None
    
    for seed in range(200):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        # Better initialization: start close to schoolbook
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize with schoolbook-like structure
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        # Add random perturbation
        U += 0.1 * np.random.randn(n*m, target_rank)
        V += 0.1 * np.random.randn(m*p, target_rank)
        W += 0.1 * np.random.randn(p*n, target_rank)
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Update U, V, W using alternating least squares
            for r in range(target_rank):
                # Update U[:,r]
                T_res = T.copy()
                for r2 in range(target_rank):
                    if r2 != r:
                        T_res -= np.einsum('a,b,c->abc', U[:,r2], V[:,r2], W[:,r2])
                
                # T_res[a,b,c] = U[a,r] * V[b,r] * W[c,r]
                # For each a, T_res[a,:,:] = U[a,r] * outer(V[:,r], W[:,r])
                # This is a rank-1 approximation
                T_res_2d = T_res.reshape(n*m, m*p*p*n)
                U_svd, s, Vt_svd = np.linalg.svd(T_res_2d, full_matrices=False)
                if len(s) > 0 and s[0] > 1e-12:
                    U[:,r] = U_svd[:,0]
                    VW = (s[0] * Vt_svd[0]).reshape(m*p, p*n)
                    V_svd, s2, Wt_svd = np.linalg.svd(VW, full_matrices=False)
                    if len(s2) > 0 and s2[0] > 1e-12:
                        V[:,r] = V_svd[:,0]
                        W[:,r] = s2[0] * Wt_svd[0]
            
            # Check error
            T_recon = np.zeros_like(T)
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            error = np.linalg.norm(T - T_recon)
            
            if error < best_error:
                best_error = error
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-10:
                break
        
        if best_error < 1e-10:
            break
    
    if best_UVW is not None and best_error < 1e-8:
        U, V, W = best_UVW
        # Try to round to rational numbers
        for denom in [1, 2, 4, 8]:
            U_rounded = np.round(U * denom) / denom
            V_rounded = np.round(V * denom) / denom
            W_rounded = np.round(W * denom) / denom
            
            T_recon = np.zeros_like(T)
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U_rounded[:,r], V_rounded[:,r], W_rounded[:,r])
            error = np.linalg.norm(T - T_recon)
            
            if error < 1e-6:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback to schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
