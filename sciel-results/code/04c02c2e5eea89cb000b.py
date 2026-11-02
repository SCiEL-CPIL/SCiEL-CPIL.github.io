"""Program for matrix multiplication tensor rank minimization."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    best_rank = n * m * p  # 56
    best_factors = _schoolbook(n, m, p, best_rank)
    
    # Try to find a lower rank decomposition
    # Known lower bound: max(n*p, m*p, n*m) = max(14, 28, 8) = 28
    target_ranks = [28, 30, 32, 36, 40, 44, 48, 52]
    
    for r in target_ranks:
        if time.time() > deadline - 30:
            break
        success, factors = _search_rank(n, m, p, r, deadline)
        if success:
            best_rank = r
            best_factors = factors
            break
    
    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p, rank):
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _search_rank(n, m, p, r, deadline):
    """Search for a rank-r decomposition of the matrix multiplication tensor."""
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Use scipy optimize to find U, V, W
    from scipy.optimize import minimize
    
    # Parameterize: flatten U, V, W into a single vector
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    size_U = n*m * r
    size_V = m*p * r
    size_W = p*n * r
    total_size = size_U + size_V + size_W
    
    def objective(x):
        U = x[:size_U].reshape(n*m, r)
        V = x[size_U:size_U+size_V].reshape(m*p, r)
        W = x[size_U+size_V:].reshape(p*n, r)
        # Compute the decomposed tensor
        # decomposed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is the outer product sum
        # Use einsum for efficiency
        decomposed = np.einsum('ar,br,cr->abc', U, V, W)
        residual = decomposed - T
        return np.sum(residual**2)
    
    # Random initialization
    np.random.seed(42)
    x0 = np.random.randn(total_size) * 0.1
    
    # Use L-BFGS-B with a time limit
    best_result = None
    remaining = deadline - time.time()
    if remaining < 5:
        return False, None
    
    # Try a few random restarts
    max_restarts = min(5, int(remaining / 10) + 1)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        np.random.seed(42 + restart * 12345)
        x0 = np.random.randn(total_size) * 0.1
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 1000, 'maxfun': 5000, 'disp': False})
        if result.fun < 1e-8:
            U = result.x[:size_U].reshape(n*m, r)
            V = result.x[size_U:size_U+size_V].reshape(m*p, r)
            W = result.x[size_U+size_V:].reshape(p*n, r)
            # Verify
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(decomposed, T, atol=1e-8):
                return True, (U, V, W)
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    
    # Check if best result is good enough
    if best_result is not None and best_result.fun < 1e-6:
        U = best_result.x[:size_U].reshape(n*m, r)
        V = best_result.x[size_U:size_U+size_V].reshape(m*p, r)
        W = best_result.x[size_U+size_V:].reshape(p*n, r)
        decomposed = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(decomposed, T, atol=1e-6):
            return True, (U, V, W)
    
    return False, None

# EVOLVE-BLOCK-END
