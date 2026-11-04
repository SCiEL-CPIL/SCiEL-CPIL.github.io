import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W, tol=1e-9):
    """
    Validate that U, V, W form an exact decomposition of the matmul(3,5,6) tensor.
    U, V, W are numpy arrays of shapes (n*m, r), (m*p, r), (p*n, r).
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != r or V.shape[1] != r or W.shape[1] != r:
        return False
    if r == 0:
        return False
    
    # For each (i,j,k), check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _alternating_minimization(rank, n, m, p, max_iterations=500, tol=1e-10, seed=None):
    """
    Alternating minimization to find a rank-r decomposition.
    Returns (U, V, W) as numpy arrays if found, None otherwise.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iterations):
        # Update U: for each (i,j), solve M @ u = ones(p)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    u_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    u_new = U[i*m + j]
                U[i*m + j] = u_new
        
        # Update V: for each (j,k), solve M @ v = ones(n)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    v_new = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    v_new = V[j*p + k]
                V[j*p + k] = v_new
        
        # Update W: for each (k,i), solve M @ w = ones(m)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    w_new = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    w_new = W[k*n + i]
                W[k*n + i] = w_new
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                    error += (val - 1.0) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        return best_U, best_V, best_W
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from just below schoolbook down to lower bound
    # But limit the number of ranks we try based on time
    ranks_to_try = list(range(schoolbook_rank - 1, lower_bound - 1, -1))
    
    # We'll try a subset of ranks, prioritizing lower ranks
    # Start from a reasonable range and go down
    # Try ranks: 89, 88, ..., down to lower_bound
    # But limit total attempts
    
    max_total_attempts = 30  # Total random restarts across all ranks
    
    attempts_used = 0
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if attempts_used >= max_total_attempts:
            break
        
        # Try multiple random initializations for this rank
        max_attempts_for_rank = 3
        found = False
        for attempt in range(max_attempts_for_rank):
            if time.time() > deadline - 5:
                break
            if attempts_used >= max_total_attempts:
                break
            
            seed = rank * 1000 + attempt
            result = _alternating_minimization(rank, n, m, p, max_iterations=300, tol=1e-10, seed=seed)
            attempts_used += 1
            
            if result is not None:
                U_np, V_np, W_np = result
                
                # Validate
                if _validate(U_np, V_np, W_np, tol=1e-9):
                    best_U, best_V, best_W = U_np, V_np, W_np
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
