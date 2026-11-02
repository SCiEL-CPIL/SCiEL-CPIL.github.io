"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _objective(params, n, m, p, rank):
    """Calculate the sum of squared errors for the tensor decomposition."""
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:size_U + size_V + size_W]
    
    U = U_flat.reshape((n*m, rank))
    V = V_flat.reshape((m*p, rank))
    W = W_flat.reshape((p*n, rank))
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                error += (s - 1.0) ** 2
    return error

def _try_rank(rank, n, m, p, max_time=10.0, seed=42):
    """Try to find a rank-rank decomposition using scipy.optimize.minimize."""
    rng = np.random.RandomState(seed)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    # Random initialization
    params = rng.randn(total_size) * 0.1
    
    start_time = time.time()
    
    # Use L-BFGS-B
    try:
        result = minimize(_objective, params, args=(n, m, p, rank), method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500})
    except Exception:
        return None
    
    if time.time() - start_time > max_time:
        return None
        
    # Check if the result is good enough
    final_error = result.fun
    if final_error < 1e-8:
        # Extract U, V, W
        U_flat = result.x[:size_U]
        V_flat = result.x[size_U:size_U + size_V]
        W_flat = result.x[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape((n*m, rank)).tolist()
        V = V_flat.reshape((m*p, rank)).tolist()
        W = W_flat.reshape((p*n, rank)).tolist()
        
        # Final verification
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
                
    # Try to reduce rank from schoolbook_rank - 1 down to 12
    for current_rank in range(schoolbook_rank - 1, 11, -1):
        if time.time() > _DEADLINE - 5:
            break
            
        found = False
        # Try a few seeds
        for seed in range(5):
            if time.time() > _DEADLINE - 5:
                break
                
            result = _try_rank(current_rank, n, m, p, max_time=10.0, seed=seed)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = current_rank
                found = True
                break
        
        if found:
            continue
        else:
            # If we failed to reduce, continue to next lower rank
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
