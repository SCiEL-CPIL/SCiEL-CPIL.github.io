"""Find a low-rank exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
_START = time.time()
_DEADLINE = 130.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=1000):
    """Try to find a decomposition with the given target rank."""
    # This is a heuristic approach: try to find U, V, W of the given rank
    # such that the tensor equation holds.
    
    # For a fixed rank r, we need to solve:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in 3*r*(n*m + m*p + p*n) variables (with nonlinear terms).
    # We'll use a simple iterative approach.
    
    # Initialize with random values
    np.random.seed(42)
    r = target_rank
    U_arr = np.random.randn(n*m, r) * 0.1
    V_arr = np.random.randn(m*p, r) * 0.1
    W_arr = np.random.randn(p*n, r) * 0.1
    
    # Scale to help with convergence
    U_arr *= 1.0 / np.sqrt(r)
    V_arr *= 1.0 / np.sqrt(r)
    W_arr *= 1.0 / np.sqrt(r)
    
    best_error = float('inf')
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
        
        # Compute current error
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    errors.append(val - 1.0)
        errors = np.array(errors)
        error = np.max(np.abs(errors))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        if error < 1e-8:
            break
        
        # Simple gradient-like update (not true gradient, but heuristic)
        # For each term, adjust U, V, W to reduce the error
        lr = 0.01
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m+j
                    idx_v = j*p+k
                    idx_w = k*n+i
                    err = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w]) - 1.0
                    
                    # Update U
                    grad_u = V_arr[idx_v] * W_arr[idx_w]
                    U_arr[idx_u] -= lr * err * grad_u
                    
                    # Update V
                    grad_v = U_arr[idx_u] * W_arr[idx_w]
                    V_arr[idx_v] -= lr * err * grad_v
                    
                    # Update W
                    grad_w = U_arr[idx_u] * V_arr[idx_v]
                    W_arr[idx_w] -= lr * err * grad_w
    
    # Check if we found a valid solution
    if best_error < 1e-7:
        # Convert to lists with rounded values (try to get integer/half-integer)
        U_list = [[round(x, 6) for x in row] for row in best_U]
        V_list = [[round(x, 6) for x in row] for row in best_V]
        W_list = [[round(x, 6) for x in row] for row in best_W]
        
        # Validate
        if _validate(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return U_list, V_list, W_list
    
    return None

def construct():
    global _START
    _START = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The theoretical minimum rank for M_{n,m,p} is at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But the actual rank is likely higher. Let's try ranks from 35 to 104.
    
    # For efficiency, try a few promising ranks
    candidate_ranks = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 104]
    
    for r in candidate_ranks:
        if _elapsed() > _DEADLINE:
            break
        
        result = _try_reduce_rank(U_school, V_school, W_school, n, m, p, r, max_iter=500)
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = r
            break  # Found a lower rank, stop
    
    # If we didn't find anything better, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
