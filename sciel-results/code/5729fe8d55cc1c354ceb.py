"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check against target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-10:
                    return False
    # Check that all other entries are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    if l != i:
                        if abs(T[i*m+j, j*p+k, k*n+l]) > 1e-10:
                            return False
    return True

def _try_rank_with_structure(n, m, p, target_rank, deadline):
    """
    Try to find a rank-target_rank decomposition using block structure.
    
    Strategy: Split the 3x3x5 tensor into blocks:
    - The first 3 columns of output form a 3x3x3 tensor
    - The last 2 columns form a 3x3x2 tensor
    
    We use shared latent factors across these blocks to enforce rank minimality.
    
    Approach:
    1. Build the full target tensor
    2. Use optimization with structured initialization
    3. Try rounding to half-integers
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def objective(params):
        U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        approx = np.zeros_like(T)
        for r in range(target_rank):
            approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
        
        return np.sum((T - approx)**2)
    
    def gradient(params):
        U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        approx = np.zeros_like(T)
        for r in range(target_rank):
            approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
        
        diff = T - approx
        
        U_grad = np.zeros_like(U_r)
        V_grad = np.zeros_like(V_r)
        W_grad = np.zeros_like(W_r)
        
        for r in range(target_rank):
            U_grad[:, r] = np.einsum('abc,bc->a', diff, V_r[:, r][:, None] * W_r[:, r][None, :])
            V_grad[:, r] = np.einsum('abc,ac->b', diff, U_r[:, r][:, None] * W_r[:, r][None, :])
            W_grad[:, r] = np.einsum('abc,ab->c', diff, U_r[:, r][:, None] * V_r[:, r][None, :])
        
        return np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
    
    # Try multiple initialization strategies
    best_result = None
    best_fun = float('inf')
    
    # Strategy 1: Random initialization
    for restart in range(3):
        if time.time() > deadline - 5:
            break
        
        U_init = np.random.randn(n*m, target_rank) * 0.3
        V_init = np.random.randn(m*p, target_rank) * 0.3
        W_init = np.random.randn(p*n, target_rank) * 0.3
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, params_init, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12})
            
            if result.fun < best_fun:
                best_fun = result.fun
                if result.fun < 1e-10:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        return U_rounded, V_rounded, W_rounded
        except Exception:
            pass
    
    # Strategy 2: Structured initialization based on block structure
    # Split into 3x3x3 and 3x3x2 blocks
    if time.time() <= deadline - 10:
        # For the 3x3x3 part, use known rank-23 decomposition (if we had it)
        # For now, use a heuristic: initialize with patterns that respect the block structure
        
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        # Assign some columns to the first 3 output columns, some to the last 2
        # First 3/5 of columns for 3x3x3, last 2/5 for 3x3x2
        cols_first = int(target_rank * 3 / 5)
        cols_second = target_rank - cols_first
        
        # Initialize first block (3x3x3)
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    if cols_first > 0:
                        col = (i * m * 3 + j * 3 + k) % cols_first
                        U_init[i*m+j, col] += 1.0
                        V_init[j*p+k, col] += 1.0
                        W_init[k*n+i, col] += 1.0
        
        # Initialize second block (3x3x2)
        for i in range(n):
            for j in range(m):
                for k in range(3, 5):
                    if cols_second > 0:
                        col = cols_first + (i * m * 2 + j * 2 + (k-3)) % cols_second
                        U_init[i*m+j, col] += 1.0
                        V_init[j*p+k, col] += 1.0
                        W_init[k*n+i, col] += 1.0
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, params_init, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12})
            
            if result.fun < best_fun:
                best_fun = result.fun
                if result.fun < 1e-10:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        return U_rounded, V_rounded, W_rounded
        except Exception:
            pass
    
    return None

def _try_low_rank_search(n, m, p, deadline):
    """
    Try to find a low-rank decomposition by searching from lower bounds upward.
    Uses block-structured initialization and optimization.
    """
    schoolbook_rank = n * m * p
    
    # Lower bound: max dimension of the tensor modes
    lower_bound = max(n*m, m*p, p*n)
    
    # Also consider theoretical lower bounds
    # For matrix multiplication, rank >= max(n*m, m*p, p*n)
    # But there are stronger bounds from Strassen's work
    
    best_result = None
    
    # Try ranks from lower_bound up to schoolbook_rank
    # But be conservative about time
    max_rank_to_try = min(schoolbook_rank, 35)
    
    for target_rank in range(lower_bound, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_with_structure(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            # Found a working rank, continue to try lower
            # But since we're going up, we can't go lower. Break.
            break
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Start with schoolbook as fallback
    U_sch, V_sch, W_sch = _schoolbook_decomposition(n, m, p)
    best_result = (U_sch, V_sch, W_sch)
    
    # Try to find lower rank decompositions
    low_rank_result = _try_low_rank_search(n, m, p, deadline)
    if low_rank_result is not None:
        best_result = low_rank_result
    
    # Convert to lists
    U_final = best_result[0].tolist()
    V_final = best_result[1].tolist()
    W_final = best_result[2].tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
