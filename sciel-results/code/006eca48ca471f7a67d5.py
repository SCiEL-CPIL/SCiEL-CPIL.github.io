import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the mean absolute error of the decomposition."""
    rank = U.shape[1]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    vals = np.einsum('ijr,jkr,kir->ijk', U_tensor, V_tensor, W_tensor)
    return np.mean(np.abs(vals - 1.0))

def _als_iteration(U, V, W, n, m, p):
    """One full ALS cycle: update U, then V, then W."""
    r = U.shape[1]
    
    # Update U: for each (i,j), solve A_mat @ u = ones(p)
    U_new = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, r))
            for k in range(p):
                A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
            b_vec = np.ones(p)
            try:
                u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U_new[i*m+j, :] = u_sol
            except:
                U_new[i*m+j, :] = U[i*m+j, :]
    U = U_new
    
    # Update V: for each (j,k), solve A_mat @ v = ones(n)
    V_new = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, r))
            for i in range(n):
                A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
            b_vec = np.ones(n)
            try:
                v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_new[j*p+k, :] = v_sol
            except:
                V_new[j*p+k, :] = V[j*p+k, :]
    V = V_new
    
    # Update W: for each (k,i), solve A_mat @ w = ones(m)
    W_new = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, r))
            for j in range(m):
                A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
            b_vec = np.ones(m)
            try:
                w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_new[k*n+i, :] = w_sol
            except:
                W_new[k*n+i, :] = W[k*n+i, :]
    W = W_new
    
    return U, V, W

def _try_als_for_rank(r, n, m, p, max_iterations=200, seed=42, time_limit=30):
    """
    Try to find a valid decomposition of rank r using ALS with random restarts.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    start_time = time.time()
    
    for restart in range(5):  # Multiple restarts
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed + restart)
        
        # Random initialization
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = _als_iteration(U, V, W, n, m, p)
            
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                return U, V, W
        
        # Check if best solution is valid
        if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-8):
            return best_U, best_V, best_W
    
    return None

def _polish(U, V, W, n, m, p, max_iterations=50, time_limit=15):
    """Final high-precision polish pass."""
    start_time = time.time()
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        
        err = _compute_error(U, V, W, n, m, p)
        if err < 1e-12:
            break
    
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find lower rank decompositions using ALS
    # Start from rank 99 and go down
    for target_rank in range(rank_school - 1, 70, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_als_for_rank(target_rank, n, m, p, max_iterations=100, seed=42, time_limit=15)
        
        if result is not None:
            U_found, V_found, W_found = result
            
            # Try to polish
            polished = _polish(U_found, V_found, W_found, n, m, p, max_iterations=30, time_limit=10)
            if polished is not None:
                U_found, V_found, W_found = polished
            
            if _verify_decomposition(U_found, V_found, W_found, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = target_rank
                break  # Found a valid decomposition, return it
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
