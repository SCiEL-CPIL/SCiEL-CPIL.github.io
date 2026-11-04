"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Use a structured decomposition based on the block structure.
    Since p=9=3*3, we can view the 3x9 matrix as three 3x3 blocks.
    
    For each 3x3 block multiplication with a 2x3 matrix, we can potentially use
    a lower-rank decomposition. The key insight is that for small rectangular
    matrix multiplication, there exist decompositions better than schoolbook.
    
    We'll use a randomized search with local optimization to find a lower-rank
    decomposition. We start with rank 40 (below schoolbook 54) and try to
    minimize the error, then verify exactness.
    
    The tensor T_{i,j,k} = delta_{j, j} for the multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We seek U (6 x r), V (27 x r), W (18 x r) such that:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if j matches, else 0
    
    More precisely, the tensor is T[i,j,k] = 1 if the indices align for multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p  # 54
    
    def schoolbook():
        rank = schoolbook_rank
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
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if i=i', j=j', k=k' else 0
    # Actually for matrix mult: C[i',k'] = sum_j A[i,j]B[j,k']
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def tensor_norm_error(U, V, W):
        """Compute ||sum_r U[:,r] V[:,r]^T W[:,r]^T - T||_F"""
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_decomp - T)
    
    def try_rank(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using random init + optimization."""
        rng = np.random.RandomState(seed)
        
        # Random initialization
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Normalize columns to help optimization
        for col in range(r):
            norm = np.sqrt(np.sum(U[:, col]**2) + np.sum(V[:, col]**2) + np.sum(W[:, col]**2))
            if norm > 1e-10:
                U[:, col] /= norm
                V[:, col] /= norm
                W[:, col] /= norm
        
        # Objective: minimize ||T_decomp - T||_F^2
        def objective(params):
            U_flat = params[:n*m*r].reshape(n*m, r)
            V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
            T_decomp = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            err = T_decomp - T
            return np.sum(err**2)
        
        # Use L-BFGS-B for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-12})
        
        U_opt = result.x[:n*m*r].reshape(n*m, r)
        V_opt = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W_opt = result.x[n*m*r + m*p*r:].reshape(p*n, r)
        
        # Check if we got close to exact
        err = tensor_norm_error(U_opt, V_opt, W_opt)
        
        # If close, try to snap to exact values
        if err < 1e-6:
            # Try rounding to nearest half-integers or integers
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            err_rounded = tensor_norm_error(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-10:
                return U_rounded, V_rounded, W_rounded, True
            
            # Try rounding to integers
            U_rounded = np.round(U_opt)
            V_rounded = np.round(V_opt)
            W_rounded = np.round(W_opt)
            err_rounded = tensor_norm_error(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-10:
                return U_rounded, V_rounded, W_rounded, True
        
        return U_opt, V_opt, W_opt, False
    
    # Try ranks from low to high
    best_result = None
    for r in range(30, min(schoolbook_rank + 1, 55)):
        if time.time() > deadline - 30:
            break
        
        # Try a few seeds for this rank
        for seed in [42, 123, 456, 789, 101112]:
            if time.time() > deadline - 30:
                break
            
            U, V, W, success = try_rank(r, max_iters=200, seed=seed)
            
            if success:
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to lists for the required interface
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
