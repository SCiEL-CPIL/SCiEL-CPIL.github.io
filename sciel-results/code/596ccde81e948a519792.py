import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _is_half_integer(x):
    """Check if x is a half-integer (multiple of 0.5) within tolerance."""
    return np.isclose(2 * x, np.round(2 * x), atol=1e-9)

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """
    Verify that U, V, W form a valid tensor decomposition for n x m x p matrix multiplication.
    """
    rank = U.shape[1]
    T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T_recon += np.einsum('a,b,c->abc', u, v, w)
    
    # Check required 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_recon[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check required 0s
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                mask[a, b, c] = True
                
    if not np.allclose(T_recon[~mask], 0.0, atol=1e-9):
        return False
        
    return True

def _check_half_ints(U, V, W):
    """Check if all entries are half-integers."""
    if not np.all(_is_half_integer(U)): return False
    if not np.all(_is_half_integer(V)): return False
    if not np.all(_is_half_integer(W)): return False
    return True

def _round_to_half_int(x):
    """Round a matrix to the nearest half-integer."""
    return np.round(2 * x) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140  # 10s margin
    
    # 1. Schoolbook (Rank 27) as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Target Tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0

    def compute_error_and_grad(params):
        """
        params: flat vector of length (n*m + m*p + p*n) * rank
        Returns: error, gradient
        """
        rank = 23 # Trying to find rank 23
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U = params[:size_U].reshape(n*m, rank)
        V = params[size_U:size_U+size_V].reshape(m*p, rank)
        W = params[size_U+size_V:].reshape(p*n, rank)
        
        # Compute T_recon
        T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        diff = T_recon - T_target
        error = np.sum(diff**2)
        
        # Gradients
        # grad_U[a, r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
        # grad_V[b, r] = 2 * sum_{a,c} diff[a,b,c] U[a,r] W[c,r]
        # grad_W[c, r] = 2 * sum_{a,b} diff[a,b,c] U[a,r] V[b,r]
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad

    # 2. Try to find Rank 23
    rank_target = 23
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize with random small values
        # Scale 1.0 is usually fine for these problems
        U_init = np.random.uniform(-1, 1, (n*m, rank_target))
        V_init = np.random.uniform(-1, 1, (m*p, rank_target))
        W_init = np.random.uniform(-1, 1, (p*n, rank_target))
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(
                compute_error_and_grad, 
                params_init, 
                method='L-BFGS-B', 
                jac=True, # Returns (f, g)
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if res.success or res.fun < 1e-6:
            # Extract optimized parameters
            params_opt = res.x
            size_U = n * m * rank_target
            size_V = m * p * rank_target
            U_opt = params_opt[:size_U].reshape(n*m, rank_target)
            V_opt = params_opt[size_U:size_U+size_V].reshape(m*p, rank_target)
            W_opt = params_opt[size_U+size_V:].reshape(p*n, rank_target)
            
            # Round to half-integers
            U_rounded = _round_to_half_int(U_opt)
            V_rounded = _round_to_half_int(V_opt)
            W_rounded = _round_to_half_int(W_opt)
            
            # Verify
            if _verify_decomposition(U_rounded, V_rounded, W_rounded):
                if _check_half_ints(U_rounded, V_rounded, W_rounded):
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = rank_target
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If rank 23 not found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
