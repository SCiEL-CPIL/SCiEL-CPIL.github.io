import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]
DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Precompute the index triples
    triples = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)
    
    # First, get the schoolbook rank-42 solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb
    
    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the tensor"""
        for a, b, c in triples:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def compute_residuals(U, V, W):
        """Compute residuals for all triples"""
        residuals = np.zeros(num_triples)
        for idx, (a, b, c) in enumerate(triples):
            residuals[idx] = np.sum(U[a, :] * V[b, :] * W[c, :]) - 1.0
        return residuals
    
    def flatten(U, V, W):
        """Flatten U, V, W into a single vector"""
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten(x):
        """Unflatten a vector into U, V, W"""
        size_U = n * m
        size_V = m * p
        size_W = p * n
        U = x[:size_U].reshape(n * m, -1)
        V = x[size_U:size_U + size_V].reshape(m * p, -1)
        W = x[size_U + size_V:].reshape(p * n, -1)
        return U, V, W
    
    def objective(x):
        """Objective function: sum of squared residuals"""
        U, V, W = unflatten(x)
        residuals = compute_residuals(U, V, W)
        return np.sum(residuals ** 2)
    
    def try_rank(R, max_iter=3000, num_restarts=5):
        """Try to find a rank-R decomposition using gradient-based optimization"""
        if R >= best_rank:
            return None
        
        best_result = None
        best_obj = np.inf
        
        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Initialize with random values
            rng = np.random.RandomState(R * 100 + restart)
            U = rng.randn(n * m, R) * 0.5
            V = rng.randn(m * p, R) * 0.5
            W = rng.randn(p * n, R) * 0.5
            
            x0 = flatten(U, V, W)
            
            # Use L-BFGS-B for optimization
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'maxfun': max_iter * 2}
                )
                
                if result.success or result.fun < best_obj:
                    best_obj = result.fun
                    U_opt, V_opt, W_opt = unflatten(result.x)
                    best_result = (U_opt, V_opt, W_opt)
                    
                    if result.fun < 1e-10:
                        return best_result
            except Exception:
                continue
        
        # Check if we found a valid solution
        if best_result is not None:
            U_try, V_try, W_try = best_result
            if verify(U_try, V_try, W_try, tol=1e-6):
                return best_result
        
        return None
    
    # Try ranks from 41 down to 21 (lower bound)
    for R in range(rank_fb - 1, 20, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_rank(R)
        if result is not None:
            U_try, V_try, W_try = result
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, R
                # Found a valid solution, continue trying lower ranks
                continue
        else:
            # No valid solution found for this rank, try next
            continue
    
    # Return the best solution found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
