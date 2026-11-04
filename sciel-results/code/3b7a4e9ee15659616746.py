"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _check_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Check if the decomposition is valid."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_error(n, m, p, U, V, W):
    """Compute the maximum error of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank(n, m, p, rank, max_iters=100, seed=42):
    """Try to find a valid decomposition with given rank."""
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Define the objective: sum of squared errors
    def objective(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U_mat = U_flat.reshape(n*m, rank)
        V_mat = V_flat.reshape(m*p, rank)
        W_mat = W_flat.reshape(p*n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                    err += (s - 1.0) ** 2
        return err
    
    # Try multiple random restarts
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_iters):
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 50})
        
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
        
        if best_error < 1e-12:
            break
    
    if best_params is None:
        return None
    
    U_mat = best_params[:n*m*rank].reshape(n*m, rank)
    V_mat = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_mat = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Check if valid
    if _check_decomposition(n, m, p, U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), tol=1e-8):
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # First, try to find a better decomposition
    # Start from rank 59 and work down
    for rank in range(n*m*p - 1, 20, -1):
        if time.time() > deadline - 30:  # Leave 30 seconds for safety
            break
        
        result = _try_rank(n, m, p, rank, max_iters=5, seed=rank)
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
