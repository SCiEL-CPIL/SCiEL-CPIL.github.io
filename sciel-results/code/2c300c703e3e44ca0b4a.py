"""Matrix multiplication 3,3,8 - rank-minimal decomposition search."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_f[i * m + j, r] * V_f[j * p + k, r] * W_f[k * n + i, r]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _alternating_least_squares(n, m, p, rank, max_iters=500, tol=1e-14, deadline=None, seed=42):
    """
    Alternating least squares to find a rank-r decomposition.
    We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    The constraint is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a system of n*m*p = 72 equations.
    
    Strategy: fix V and W, solve for U via linear least squares, then fix U and W, solve for V,
    then fix U and V, solve for W, iterate.
    """
    rng = np.random.RandomState(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Initialize randomly
    U = rng.randn(n_m, rank) * 0.1
    V = rng.randn(m_p, rank) * 0.1
    W = rng.randn(p_n, rank) * 0.1
    
    # Precompute index mappings
    # For each (i, j, k), we have:
    #   U_idx = i * m + j
    #   V_idx = j * p + k
    #   W_idx = k * n + i
    
    # Build the system for solving U given V, W:
    # For each (i, j, k): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    # This is: for fixed (i,j), sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
    
    # Actually, let's think of it differently. The constraint is:
    # For all (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # If we fix V and W, then for each (i, j), we have p equations:
    #   sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1, for k = 0, ..., p-1
    # This is a linear system in U[i*m+j, :] (which has rank components).
    
    # Similarly for V and W.
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            return None
        
        # Solve for U: for each (i, j), solve A_U @ U[i*m+j, :] = b_U
        # where A_U[k, r] = V[j*p+k, r] * W[k*n+i, r] and b_U[k] = 1
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build the system
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                # Solve least squares
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                U_new[i * m + j, :] = sol
        
        # Solve for V: for each (j, k), solve A_V @ V[j*p+k, :] = b_V
        # where A_V[i, r] = U[i*m+j, r] * W[k*n+i, r] and b_V[i] = 1
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                V_new[j * p + k, :] = sol
        
        # Solve for W: for each (k, i), solve A_W @ W[k*n+i, :] = b_W
        # where A_W[j, r] = U[i*m+j, r] * V[j*p+k, r] and b_W[j] = 1
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                W_new[k * n + i, :] = sol
        
        # Check convergence
        diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        
        if diff < tol:
            break
    
    # Verify the result
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_als(n, m, p, target_rank, num_trials=50, deadline=None):
    """Try alternating least squares for a given target rank."""
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
        seed = trial * 7919 + 42
        result = _alternating_least_squares(n, m, p, target_rank, 
                                            max_iters=300, tol=1e-14,
                                            deadline=deadline, seed=seed)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, target_rank):
                return {"U": U, "V": V, "W": W}
    return None

def _try_rank_gradient(n, m, p, target_rank, num_trials=20, deadline=None):
    """Try gradient-based optimization for a given target rank."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Total number of variables
    n_vars = (n_m + m_p + p_n) * target_rank
    
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
        
        seed = trial * 104729 + 13
        rng = np.random.RandomState(seed)
        
        # Initialize
        x0 = rng.randn(n_vars) * 0.1
        
        def objective(x):
            if deadline and time.time() > deadline:
                return 1e10
            U = x[:n_m * target_rank].reshape(n_m, target_rank)
            V = x[n_m * target_rank:n_m * target_rank + m_p * target_rank].reshape(m_p, target_rank)
            W = x[n_m * target_rank + m_p * target_rank:].reshape(p_n, target_rank)
            
            # Compute sum of squared errors
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            return err
        
        # Try to minimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-15})
            if res.success and res.fun < 1e-10:
                U = res.x[:n_m * target_rank].reshape(n_m, target_rank)
                V = res.x[n_m * target_rank:n_m * target_rank + m_p * target_rank].reshape(m_p, target_rank)
                W = res.x[n_m * target_rank + m_p * target_rank:].reshape(p_n, target_rank)
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    
    # Theoretical lower bound: at least n*p = 24 (each output needs at least one term)
    # Actually, the lower bound is more nuanced. For matmul(n,m,p), the rank is at least
    # max(n*p, m*p, n*m) in some cases, but for the standard matmul tensor, it's known
    # to be n*m*p in general position.
    
    # Try ranks from 24 upward
    for target_rank in range(24, best_rank):
        if time.time() > deadline:
            break
        
        # Try ALS first (faster)
        result = _try_rank_als(n, m, p, target_rank, num_trials=30, deadline=deadline)
        if result is not None:
            best = result
            best_rank = target_rank
            break
        
        # Try gradient-based (slower but more powerful)
        if time.time() < deadline - 30:
            result = _try_rank_gradient(n, m, p, target_rank, num_trials=10, deadline=deadline)
            if result is not None:
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
