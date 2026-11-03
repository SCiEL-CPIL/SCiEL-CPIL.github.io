import time
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 120
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb
    
    def check_decomp(U, V, W, atol=1e-6):
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=atol)
    
    def snap_to_half(X):
        return np.round(X * 2) / 2
    
    def try_rank_als(rank, max_iters=200, seed=42):
        size_u, size_v, size_w = n*m, m*p, p*n
        
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(size_u, rank) * 0.05
        V = np.random.randn(size_v, rank) * 0.05
        W = np.random.randn(size_w, rank) * 0.05
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for a in range(size_u):
                M = np.einsum('bt,ct->bct', V, W).reshape(size_v * size_w, rank)
                b_vec = T[a, :, :].reshape(-1)
                try:
                    U[a, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(size_v):
                M = np.einsum('at,ct->act', U, W).reshape(size_u * size_w, rank)
                b_vec = T[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(size_w):
                M = np.einsum('at,bt->abt', U, V).reshape(size_u * size_v, rank)
                b_vec = T[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Compute error
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            err = np.sum((T - T_decomp) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-14:
                break
        
        # Try to snap the best solution
        U_s = snap_to_half(best_U)
        V_s = snap_to_half(best_V)
        W_s = snap_to_half(best_W)
        if check_decomp(U_s, V_s, W_s, atol=1e-5):
            return U_s, V_s, W_s
        
        # Try snapping the current solution
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if check_decomp(U_s, V_s, W_s, atol=1e-5):
            return U_s, V_s, W_s
        
        # Check if current is close to half-integers and valid
        if check_decomp(U, V, W, atol=1e-8):
            if np.allclose(U, U_s, atol=1e-2) and np.allclose(V, V_s, atol=1e-2) and np.allclose(W, W_s, atol=1e-2):
                return U_s, V_s, W_s
        
        return None
    
    # Try rank 32 with multiple seeds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333, 444, 555]
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        result = try_rank_als(32, max_iters=150, seed=seed)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = 32
            break
    
    if best_rank == 32:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
