import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1
    return T

def _verify_exact(U, V, W, T, tol=1e-9):
    """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T exactly (within float tol)."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _project_half_integer(M):
    """Project matrix entries to nearest half-integer."""
    return np.round(M * 2.0) / 2.0

def _als_step_fix_UV(U, V, T, n, m, p):
    """Fix U and V, solve for W to best fit T.
    T ≈ sum_r U_r ⊗ V_r ⊗ W_r
    For each gamma: T[:,:,gamma] ≈ sum_r (U_r ⊗ V_r) * W_r[gamma]
    """
    rank = U.shape[1]
    # Build the coefficient matrix: for each gamma, we have a linear system
    # Actually, we can vectorize: 
    # T[alpha, beta, gamma] = sum_r U[alpha,r] * V[beta,r] * W[gamma,r]
    # For fixed U,V: for each gamma, T[:,:,gamma] = sum_r W[gamma,r] * (U[:,r] * V[:,r])
    # So T[:,:,gamma] = (U * V) @ W[gamma,:]  where U*V is element-wise outer per r
    # Let's form: for each r, the matrix M_r = U[:,r] * V[:,r] (outer product, n*m x m*p)
    # T[:,:,gamma] = sum_r M_r * W[gamma,r]
    # Reshape: T_flat[gamma, alpha*m*p+beta] = sum_r W[gamma,r] * M_r_flat[alpha*m*p+beta]
    # So T_flat = W^T @ M_flat where M_flat has shape (rank, n*m*m*p)
    
    M_flat = np.zeros((rank, n * m * m * p))
    for r in range(rank):
        M_flat[r] = (U[:, r] * V[:, r]).flatten()
    
    T_flat = T.reshape(p * n, n * m * m * p).T  # shape (n*m*m*p, p*n)
    # We want: T_flat = M_flat @ W_flat where W_flat has shape (rank, p*n)
    # So W_flat = pinv(M_flat) @ T_flat
    try:
        W_flat = np.linalg.lstsq(M_flat, T_flat, rcond=None)[0]
    except:
        W_flat = np.zeros((rank, p * n))
    W = W_flat.T  # shape (p*n, rank)
    return W

def _als_step_fix_VW(V, W, T, n, m, p):
    """Fix V and W, solve for U."""
    rank = V.shape[1]
    # T[alpha, beta, gamma] = sum_r U[alpha,r] * V[beta,r] * W[gamma,r]
    # For fixed V,W: for each alpha, T[alpha,:,:] = sum_r U[alpha,r] * (V[:,r] * W[:,r])
    # M_r = V[:,r] * W[:,r], shape (m*p, p*n)
    # T[alpha, beta, gamma] = sum_r U[alpha,r] * M_r[beta, gamma]
    # T_flat[alpha, beta*p*n+gamma] = sum_r U[alpha,r] * M_flat[r, beta*p*n+gamma]
    # T_flat = U^T @ M_flat where T_flat shape (n*m, m*p*p*n), M_flat shape (rank, m*p*p*n)
    
    M_flat = np.zeros((rank, m * p * p * n))
    for r in range(rank):
        M_flat[r] = (V[:, r] * W[:, r]).flatten()
    
    T_flat = T.reshape(n * m, m * p * p * n)  # shape (n*m, m*p*p*n)
    # T_flat = U^T @ M_flat => U = pinv(M_flat) @ T_flat^T
    try:
        U = np.linalg.lstsq(M_flat, T_flat.T, rcond=None)[0].T
    except:
        U = np.zeros((n * m, rank))
    return U

def _als_step_fix_UW(U, W, T, n, m, p):
    """Fix U and W, solve for V."""
    rank = U.shape[1]
    # T[alpha, beta, gamma] = sum_r U[alpha,r] * V[beta,r] * W[gamma,r]
    # For fixed U,W: for each beta, T[:,beta,:] = sum_r V[beta,r] * (U[:,r] * W[:,r])
    # M_r = U[:,r] * W[:,r], shape (n*m, p*n)
    # T_flat[beta, alpha*p*n+gamma] = sum_r V[beta,r] * M_flat[r, alpha*p*n+gamma]
    # T_flat = V^T @ M_flat where T_flat shape (m*p, n*m*p*n), M_flat shape (rank, n*m*p*n)
    
    M_flat = np.zeros((rank, n * m * p * n))
    for r in range(rank):
        M_flat[r] = (U[:, r] * W[:, r]).flatten()
    
    T_flat = T.reshape(m * p, n * m * p * n)  # Wait, need to check indexing
    # T[alpha, beta, gamma], alpha in [0,n*m), beta in [0,m*p), gamma in [0,p*n)
    # For fixed beta: T[:,beta,:] has shape (n*m, p*n)
    # So T_flat should be indexed as [beta, alpha*p*n + gamma]
    T_flat = np.zeros((m * p, n * m * p * n))
    for beta in range(m * p):
        for alpha in range(n * m):
            for gamma in range(p * n):
                T_flat[beta, alpha * p * n + gamma] = T[alpha, beta, gamma]
    
    try:
        V = np.linalg.lstsq(M_flat, T_flat.T, rcond=None)[0].T
    except:
        V = np.zeros((m * p, rank))
    return V

def _try_rank_als(T, rank, n, m, p, max_iters=50, deadline=None):
    """Try to find exact rank-r decomposition via ALS."""
    for attempt in range(10):
        if deadline and time.time() > deadline - 10:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline and time.time() > deadline - 10:
                break
            
            # Alternating least squares
            W = _als_step_fix_UV(U, V, T, n, m, p)
            U = _als_step_fix_VW(V, W, T, n, m, p)
            V = _als_step_fix_UW(U, W, T, n, m, p)
            
            # Check error
            err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Try half-integer projection and check exactness
            U_h = _project_half_integer(best_U)
            V_h = _project_half_integer(best_V)
            W_h = _project_half_integer(best_W)
            
            if _verify_exact(U_h, V_h, W_h, T):
                return U_h, V_h, W_h
        
        # Final check on best
        U_h = _project_half_integer(best_U)
        V_h = _project_half_integer(best_V)
        W_h = _project_half_integer(best_W)
        if _verify_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
    
    return None

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    T = _build_target_tensor()
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank via ALS with half-integer projection
    # Try ranks from a reasonable lower bound up
    # Known lower bounds for 5x5x5 matmul rank are around 90+
    # Try a range of ranks
    for rank in range(100, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_als(T, rank, n, m, p, max_iters=30, deadline=deadline)
        if result is not None:
            U_h, V_h, W_h = result
            best_rank = rank
            best_U, best_V, best_W = U_h, V_h, W_h
            break  # Found a valid decomposition at this rank
    
    # Also try a few specific structured ranks
    for rank in [95, 90, 85, 80, 75, 70, 65, 60]:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
        
        result = _try_rank_als(T, rank, n, m, p, max_iters=20, deadline=deadline)
        if result is not None:
            U_h, V_h, W_h = result
            best_rank = rank
            best_U, best_V, best_W = U_h, V_h, W_h
    
    # Return the best found
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
