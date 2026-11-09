"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _compute_sse_and_grad(params, n, m, p, rank):
    """Compute SSE and gradient for tensor decomposition."""
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
    W = params[(n*m + m*p)*rank:].reshape(p*n, rank)
    
    sse = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i*m+j]
                v = V[j*p+k]
                w = W[k*n+i]
                val = np.dot(u, v * w)
                err = val - 1.0
                sse += err * err
                
                # Gradient: d/dU = 2*err * v*w, d/dV = 2*err * u*w, d/dW = 2*err * u*v
                grad_U[i*m+j] += 2 * err * v * w
                grad_V[j*p+k] += 2 * err * u * w
                grad_W[k*n+i] += 2 * err * u * v
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return sse, grad

def _verify_solution(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify if the solution is exact (within tolerance) and has integer/half-integer entries."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > tol:
        return False
    
    # Check for integer/half-integer entries
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if close to integer
                if abs(val - round(val)) > 1e-7:
                    # Check if close to half-integer
                    if abs(val * 2 - round(val * 2)) > 1e-7:
                        return False
    return True

def _round_to_half(x):
    """Round array to nearest 0.5."""
    return np.round(x * 2) / 2

def _try_rank(n, m, p, rank, time_limit=5.0):
    """Try to find a rank-rank solution. Returns (U, V, W) or None."""
    start_time = time.time()
    n_params = (n*m + m*p + p*n) * rank
    
    # Try multiple random initializations with different strategies
    for attempt in range(15):
        if time.time() - start_time > time_limit:
            break
        
        # Different initialization strategies
        if attempt < 5:
            # Small random
            np.random.seed(attempt * 12345 + rank)
            params = np.random.randn(n_params) * 0.3
        elif attempt < 10:
            # Larger random with some structure
            np.random.seed(attempt * 7777 + rank)
            params = np.random.randn(n_params) * 0.8
        else:
            # Sparse-ish initialization
            np.random.seed(attempt * 9999 + rank)
            params = np.random.randn(n_params) * 0.5
            # Zero out some entries to encourage sparsity
            mask = np.random.rand(n_params) > 0.5
            params[mask] *= 0.1
        
        # Optimize using L-BFGS-B which handles gradients well
        def objective(x):
            sse, _ = _compute_sse_and_grad(x, n, m, p, rank)
            return sse
        
        def grad(x):
            _, g = _compute_sse_and_grad(x, n, m, p, rank)
            return g
        
        try:
            result = minimize(objective, params, jac=grad, method='L-BFGS-B', 
                             options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if result.fun < 1e-6:
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
                W = result.x[(n*m + m*p)*rank:].reshape(p*n, rank)
                
                # Try rounding to nearest 0.5
                U_rounded = _round_to_half(U)
                V_rounded = _round_to_half(V)
                W_rounded = _round_to_half(W)
                
                if _verify_solution(U_rounded, V_rounded, W_rounded, n, m, p, rank):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # Check if unrounded is valid
                    if _verify_solution(U, V, W, n, m, p, rank):
                        return U, V, W
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    max_rank = n * m * p
    
    # Start with the trivial solution (Rank 36)
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_result = (U, V, W)
    best_rank = max_rank
    
    # Try to find a lower rank
    # Start from rank 35 and go down. Use more time for higher ranks first.
    for rank in range(max_rank - 1, 20, -1):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate more time to higher ranks (closer to trivial solution)
        time_fraction = remaining / (deadline - start_time)
        time_limit = min(8.0, max(2.0, time_fraction * 8.0))
        
        result = _try_rank(n, m, p, rank, time_limit=time_limit)
        
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break  # Found a lower rank, stop
    
    U, V, W = best_result
    
    # Convert to list of lists with proper types
    def to_clean_list(mat):
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-7:
                    clean_row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-7:
                    clean_row.append(val * 2 / 2)  # half-integer
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
# EVOLVE-BLOCK-END
