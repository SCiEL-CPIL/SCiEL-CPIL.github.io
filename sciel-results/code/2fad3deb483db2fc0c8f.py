"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_tensor_error(n, m, p, U_mat, V_mat, W_mat):
    """Compute sum of squared errors using vectorized operations."""
    rank = U_mat.shape[1]
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum over rank: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                prod = U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :]
                s = np.sum(prod)
                total_error += (s - 1.0) ** 2
    return total_error

def _check_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Check if the decomposition is valid."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, rank, max_restarts=20, seed=42, time_limit=10.0):
    """Try to find a valid decomposition with given rank."""
    start_time = time.time()
    np.random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    total_params = nm * rank + mp * rank + pn * rank
    
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random values
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Define the objective: sum of squared errors
        def objective(params_vec):
            U_mat = params_vec[:nm*rank].reshape(nm, rank)
            V_mat = params_vec[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
            W_mat = params_vec[nm*rank + mp*rank:].reshape(pn, rank)
            
            return _compute_tensor_error(n, m, p, U_mat, V_mat, W_mat)
        
        res = minimize(objective, params, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x.copy()
        
        if best_error < 1e-10:
            break
    
    if best_params is None or best_error > 1e-8:
        return None
    
    U_mat = best_params[:nm*rank].reshape(nm, rank)
    V_mat = best_params[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
    W_mat = best_params[nm*rank + mp*rank:].reshape(pn, rank)
    
    # Final check
    if _check_decomposition(n, m, p, U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), tol=1e-7):
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None

def _try_strassen_blocks(n, m, p):
    """
    Try a block decomposition approach using Strassen-like blocks.
    For 3x4x5, we can try various block decompositions.
    """
    # This is a placeholder for more sophisticated block approaches
    # The current optimization should handle this
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # Leave 10 seconds safety margin
    
    # Try to find a better decomposition
    # Start from lower ranks and work up, since lower rank is harder
    # but we want the minimum rank
    
    # First try a range of ranks from low to high
    min_rank = 20  # Lower bound estimate
    max_rank = n * m * p  # 60
    
    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline - 20:  # Leave 20 seconds for safety
            break
        
        # Try multiple seeds for each rank
        for seed in [rank, rank + 100, rank + 200]:
            if time.time() > deadline - 20:
                break
            
            result = _try_rank(n, m, p, rank, max_restarts=15, seed=seed, time_limit=8.0)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
