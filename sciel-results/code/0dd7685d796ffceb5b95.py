import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _als_with_init(T, rank, init_U, init_V, init_W, max_iters=500, tol=1e-12):
    """
    ALS with initialization.
    """
    dim = 9
    U = init_U.copy()
    V = init_V.copy()
    W = init_W.copy()
    
    prev_error = float('inf')
    
    for it in range(max_iters):
        # Update U
        for a in range(dim):
            M = np.einsum('br,cr->bcr', V, W).reshape(dim*dim, rank)
            T_flat = T[a].reshape(-1)
            U[a], _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            
        # Update V
        for b in range(dim):
            M = np.einsum('ar,cr->acr', U, W).reshape(dim*dim, rank)
            T_flat = T[:, b].reshape(-1)
            V[b], _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            
        # Update W
        for c in range(dim):
            M = np.einsum('ar,br->abr', U, V).reshape(dim*dim, rank)
            T_flat = T[:, :, c].reshape(-1)
            W[c], _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            
        # Error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-7:
                return U, V, W
            else:
                return None, None, None
        prev_error = error
        
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build Target Tensor
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Get schoolbook decomposition for initialization
    U0, V0, W0 = _schoolbook(n, m, p)
    
    attempts = 0
    max_attempts = 300 # 300 attempts. Each is fast.
    
    random.seed(42)
    
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Randomly select 4 indices to drop from the 27 schoolbook terms
        drop_indices = random.sample(range(27), 4)
        keep_indices = [i for i in range(27) if i not in drop_indices]
        
        init_U = U0[:, keep_indices]
        init_V = V0[:, keep_indices]
        init_W = W0[:, keep_indices]
        
        # Add small noise to break symmetry and help escape local minima
        init_U = init_U + np.random.randn(dim_a, 23) * 0.1
        init_V = init_V + np.random.randn(dim_b, 23) * 0.1
        init_W = init_W + np.random.randn(dim_c, 23) * 0.1
        
        U, V, W = _als_with_init(T, 23, init_U, init_V, init_W, max_iters=500)
        
        if U is not None:
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to Schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
