import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, tol=1e-9):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_half(arr):
    """Snap to nearest half-integer."""
    return np.round(np.array(arr, dtype=float) * 2) / 2.0

def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 90."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return U, V, W

def _compute_error(U, V, W):
    """Compute the sum of squared errors."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                err += (val - 1.0) ** 2
    return err

def _alternating_minimization(rank, n, m, p, max_iterations=2000, tol=1e-10, time_limit=None, start_time=None):
    """
    Alternating least squares to find a rank-r decomposition.
    At each step, solve linear subproblems exactly.
    """
    deadline = start_time + time_limit if time_limit else float('inf')
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline - 2:
            break
        
        # Random initialization with reasonable scale
        scale = 1.0 / (rank ** 0.33)
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        for it in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # Build matrix M where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p + k] * W[k*n + i]
                    try:
                        u_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        u_new = U[i*m + j]
                    U[i*m + j] = u_new
            
            # Update V: for each (j,k), solve linear system
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
            
            # Update W: for each (k,i), solve linear system
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
            err = _compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < tol:
                break
        
        if best_err < tol:
            break
    
    if best_err < 1e-8 and best_U is not None:
        # Snap to half-integers and verify
        U_snap = _snap_half(best_U)
        V_snap = _snap_half(best_V)
        W_snap = _snap_half(best_W)
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list, best_err
    
    return None, None, None, best_err

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions using alternating minimization
    # Lower bound is max(n*m, m*p, p*n) = 30
    # Try a range of ranks, spending more time on lower ranks
    ranks_to_try = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 88, 89]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            break
        
        # Allocate time based on rank - lower ranks get more time
        time_alloc = min(10, max(3, 20 - (rank - 30) * 0.4))
        
        U_res, V_res, W_res, err = _alternating_minimization(
            rank, n, m, p, 
            max_iterations=1500, 
            tol=1e-10,
            time_limit=time_alloc,
            start_time=start_time
        )
        
        if U_res is not None:
            best_U, best_V, best_W = U_res, V_res, W_res
            best_rank = rank
            # If we found a good rank, we can stop early
            if rank <= 35:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
