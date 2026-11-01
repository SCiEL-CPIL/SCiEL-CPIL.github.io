"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np
from scipy.optimize import minimize

def _error_function(params, n, m, p, rank):
    """Calculate the error of the decomposition."""
    U = params[:n*m*rank].reshape((n*m, rank))
    V = params[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
    W = params[n*m*rank + m*p*rank :].reshape((p*n, rank))
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    # Target is 1 if j == jp, else 0
                    target = 1.0 if j == jp else 0.0
                    # Sum over c
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                    error += (val - target) ** 2
    return error

def _check_decomposition(U, V, W, n, m, p, rank):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = np.sum(U[i*m + j, :] * V[jp*p + k, :] * W[k*n + i, :])
                    if abs(val - target) > 1e-6:
                        return False
    return True

def _try_find_decomposition(n, m, p, rank, max_iters=1000):
    """Try to find a rank-r decomposition using optimization."""
    if time.time() > DEADLINE - SAFETY_MARGIN:
        return None
        
    total_vars = n*m*rank + m*p*rank + p*n*rank
    # Initialize with small random values
    np.random.seed(42)
    x0 = np.random.uniform(-1, 1, size=total_vars)
    
    # Define objective
    def obj(x):
        return _error_function(x, n, m, p, rank)
    
    # Try to minimize
    # Use a method that handles the non-convexity somewhat
    result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if result.fun < 1e-4:
        # Extract U, V, W
        U = result.x[:n*m*rank].reshape((n*m, rank))
        V = result.x[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
        W = result.x[n*m*rank + m*p*rank :].reshape((p*n, rank))
        
        # Round to nearest integer or half-integer?
        # The problem allows integer/half-integer.
        # Let's try rounding to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _check_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, rank):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
        # If rounding failed, check if the raw float solution is "exact" enough
        # and maybe it's an integer solution that didn't round well?
        if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
             # Try to snap to integers
             U_i = np.round(U)
             V_i = np.round(V)
             W_i = np.round(W)
             if _check_decomposition(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p, rank):
                 return U_i.tolist(), V_i.tolist(), W_i.tolist()
             return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from 13 upwards
    for rank in range(13, schoolbook_rank + 1):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
            
        result = _try_find_decomposition(n, m, p, rank, max_iters=200)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fall back to schoolbook
    rank = schoolbook_rank
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
