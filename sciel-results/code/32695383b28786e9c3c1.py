import numpy as np
import time

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Generate the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1
    
    def als_decompose(r, max_iter=2000, tol=1e-14, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        prev_residual = np.inf
        stagnant_count = 0
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            VW = (V[:, None, :] * W[None, :, :]).reshape(m*p * p*n, r)
            for a in range(n*m):
                U[a, :] = np.linalg.lstsq(VW, T[a, :, :].flatten(), rcond=None)[0]
            
            # Update V
            UW = (U[:, None, :] * W[None, :, :]).reshape(n*m * p*n, r)
            for b in range(m*p):
                V[b, :] = np.linalg.lstsq(UW, T[:, b, :].flatten(), rcond=None)[0]
            
            # Update W
            UV = (U[:, None, :] * V[None, :, :]).reshape(n*m * m*p, r)
            for c in range(p*n):
                W[c, :] = np.linalg.lstsq(UV, T[:, :, c].flatten(), rcond=None)[0]
            
            # Check convergence
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T_hat - T)
            
            if residual < tol:
                break
            
            if residual > prev_residual * 0.999:
                stagnant_count += 1
            else:
                stagnant_count = 0
            prev_residual = residual
            
            if stagnant_count > 200:
                break
        
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_hat - T)
        return U, V, W, residual
    
    # Try to find a lower-rank decomposition
    best_rank = 21
    best_U, best_V, best_W = None, None, None
    
    for r in [20, 19, 18, 17]:
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(15):
            if time.time() > deadline:
                break
            
            U, V, W, residual = als_decompose(r, max_iter=1500, seed=trial + 100)
            
            if residual < 1e-10:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    if best_rank < 21:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: rank 21 Strassen
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    num_blocks = p // 2
    for k in range(num_blocks):
        base = k * 7
        U[:, base:base+7] = U_local
        V[2*k, base:base+7] = V_local[0]
        V[2*k+1, base:base+7] = V_local[1]
        V[6+2*k, base:base+7] = V_local[2]
        V[7+2*k, base:base+7] = V_local[3]
        W[4*k, base:base+7] = W_local[0]
        W[4*k+2, base:base+7] = W_local[1]
        W[4*k+1, base:base+7] = W_local[2]
        W[4*k+3, base:base+7] = W_local[3]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
