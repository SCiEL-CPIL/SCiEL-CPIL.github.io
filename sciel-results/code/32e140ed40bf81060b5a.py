import numpy as np
import time

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Generate the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1
    
    def compute_residual(U, V, W):
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_hat - T)
    
    def als_decompose(r, max_iter=2000, tol=1e-12, seed=0, init_U=None, init_V=None, init_W=None):
        if init_U is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
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
            residual = compute_residual(U, V, W)
            
            if residual < tol:
                break
            
            if residual > prev_residual * 0.999:
                stagnant_count += 1
            else:
                stagnant_count = 0
            prev_residual = residual
            
            if stagnant_count > 100:
                break
        
        residual = compute_residual(U, V, W)
        return U, V, W, residual
    
    # Build the rank 21 Strassen solution
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
    
    rank_21 = 21
    U_21 = np.zeros((n*m, rank_21))
    V_21 = np.zeros((m*p, rank_21))
    W_21 = np.zeros((p*n, rank_21))
    
    num_blocks = p // 2
    for k in range(num_blocks):
        base = k * 7
        U_21[:, base:base+7] = U_local
        V_21[2*k, base:base+7] = V_local[0]
        V_21[2*k+1, base:base+7] = V_local[1]
        V_21[6+2*k, base:base+7] = V_local[2]
        V_21[7+2*k, base:base+7] = V_local[3]
        W_21[4*k, base:base+7] = W_local[0]
        W_21[4*k+2, base:base+7] = W_local[1]
        W_21[4*k+1, base:base+7] = W_local[2]
        W_21[4*k+3, base:base+7] = W_local[3]
    
    best_rank = 21
    best_U, best_V, best_W = U_21, V_21, W_21
    
    # Strategy 1: Try to find rank 20 by removing one term from rank 21
    for term_to_remove in range(21):
        if time.time() > deadline:
            break
        
        U_init = np.delete(U_21, term_to_remove, axis=1)
        V_init = np.delete(V_21, term_to_remove, axis=1)
        W_init = np.delete(W_21, term_to_remove, axis=1)
        
        # Add small noise to break symmetry
        np.random.seed(term_to_remove + 42)
        U_init += np.random.randn(n*m, 20) * 0.01
        V_init += np.random.randn(m*p, 20) * 0.01
        W_init += np.random.randn(p*n, 20) * 0.01
        
        U, V, W, residual = als_decompose(20, max_iter=1500, tol=1e-12, init_U=U_init, init_V=V_init, init_W=W_init)
        
        if residual < 1e-10:
            best_rank = 20
            best_U, best_V, best_W = U, V, W
            break
    
    # Strategy 2: Random restarts for rank 20
    if best_rank == 21:
        for trial in range(100):
            if time.time() > deadline:
                break
            U, V, W, residual = als_decompose(20, max_iter=1500, tol=1e-12, seed=trial + 1000)
            if residual < 1e-10:
                best_rank = 20
                best_U, best_V, best_W = U, V, W
                break
    
    # Strategy 3: Try rank 19 if time permits
    if best_rank > 19 and time.time() < deadline - 30:
        for trial in range(50):
            if time.time() > deadline:
                break
            U, V, W, residual = als_decompose(19, max_iter=1000, tol=1e-12, seed=trial + 2000)
            if residual < 1e-10:
                best_rank = 19
                best_U, best_V, best_W = U, V, W
                break
    
    if best_rank < 21:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U_21.tolist(), "V": V_21.tolist(), "W": W_21.tolist()}
