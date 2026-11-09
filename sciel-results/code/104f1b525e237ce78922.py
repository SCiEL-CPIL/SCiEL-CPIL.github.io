"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _compute_sse(params, n, m, p, rank):
    """Compute the sum of squared errors of the tensor decomposition."""
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
    W = params[(n*m + m*p)*rank:].reshape(p*n, rank)
    
    sse = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                sse += (val - 1.0)**2
    return sse

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
                if abs(val - round(val)) > 1e-7:
                    if abs(val * 2 - round(val * 2)) > 1e-7:
                        return False
    return True

def _try_rank(n, m, p, rank, time_limit=5.0):
    """Try to find a rank-rank solution. Returns (U, V, W) or None."""
    start_time = time.time()
    n_params = (n*m + m*p + p*n) * rank
    
    for attempt in range(25):
        if time.time() - start_time > time_limit:
            break
        
        # Multiple initialization strategies
        if attempt < 8:
            np.random.seed(attempt * 12345 + rank)
            params = np.random.randn(n_params) * 0.4
        elif attempt < 16:
            # Sparse initialization
            np.random.seed(attempt * 777 + rank)
            params = np.zeros(n_params)
            for c in range(rank):
                for idx in range(n*m):
                    if np.random.random() < 0.4:
                        params[idx*rank + c] = np.random.choice([-1, 1]) * 0.5
                for idx in range(m*p):
                    if np.random.random() < 0.4:
                        params[(n*m + idx)*rank + c] = np.random.choice([-1, 1]) * 0.5
                for idx in range(p*n):
                    if np.random.random() < 0.4:
                        params[(n*m + m*p + idx)*rank + c] = np.random.choice([-1, 1]) * 0.5
        else:
            # Near-trivial with perturbation
            np.random.seed(attempt * 555 + rank)
            params = np.zeros(n_params)
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                params[(i*m + j)*rank + c] = 1.0
                params[(n*m + j*p + k)*rank + c] = 1.0
                params[(n*m + m*p + k*n + i)*rank + c] = 1.0
            params += np.random.randn(n_params) * 0.03
        
        def objective(x):
            return _compute_sse(x, n, m, p, rank)
        
        try:
            result = minimize(objective, params, method='BFGS', 
                             options={'maxiter': 400, 'gtol': 1e-9})
            
            if result.fun < 1e-7:
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
                W = result.x[(n*m + m*p)*rank:].reshape(p*n, rank)
                
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify_solution(U_rounded, V_rounded, W_rounded, n, m, p, rank):
                    return U_rounded, V_rounded, W_rounded
                if _verify_solution(U, V, W, n, m, p, rank):
                    return U, V, W
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
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
    
    # Try ranks from 35 down to 20
    for rank in range(max_rank - 1, 19, -1):
        if time.time() > deadline - 10:
            break
        
        time_limit = min(6.0, max(1.5, deadline - time.time() - 10))
        result = _try_rank(n, m, p, rank, time_limit=time_limit)
        
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break
    
    U, V, W = best_result
    
    def to_clean_list(mat):
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-7:
                    clean_row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-7:
                    clean_row.append(float(int(val * 2)) / 2.0)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
# EVOLVE-BLOCK-END
