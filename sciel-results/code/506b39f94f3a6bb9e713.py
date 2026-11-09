import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact. U, V, W are numpy arrays."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition."""
    err_sq = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m + j, :]
            for k in range(p):
                v_row = V[j*p + k, :]
                w_row = W[k*n + i, :]
                s = np.sum(u_row * v_row * w_row)
                err_sq += (s - 1.0)**2
    return err_sq

def _try_optimize_rank(n, m, p, target_rank, time_limit, max_restarts=15):
    """
    Try to find a decomposition of rank target_rank using optimization.
    Returns (U, V, W) if found and verified, else None.
    """
    for restart in range(max_restarts):
        if time.time() > time_limit:
            break
        
        # Initialize with random values, scaled appropriately
        scale = 1.0 / (target_rank ** 0.5)
        U = np.random.randn(n * m, target_rank) * scale
        V = np.random.randn(m * p, target_rank) * scale
        W = np.random.randn(p * n, target_rank) * scale
        
        def unflatten(params):
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            U = U_flat.reshape(n * m, target_rank)
            V = V_flat.reshape(m * p, target_rank)
            W = W_flat.reshape(p * n, target_rank)
            return U, V, W
        
        def cost(params):
            U, V, W = unflatten(params)
            return _compute_error(U, V, W, n, m, p)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B
        result = minimize(cost, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-8})
        
        if result.fun < 1e-10:
            U, V, W = unflatten(result.x)
            if _verify(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions, starting from rank 59
    for target_rank in range(best_rank - 1, 45, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_optimize_rank(n, m, p, target_rank, deadline - 10, max_restarts=8)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
