import time
import numpy as np
from itertools import product

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

def _snap_quarter(arr):
    """Snap to nearest quarter-integer."""
    return np.round(np.array(arr, dtype=float) * 4) / 4.0

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

def _structured_init(rank, n, m, p, seed=0):
    """
    Structured initialization exploiting the matmul tensor structure.
    Uses patterns based on the index relationships.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy 1: Block-diagonal-like structure
    # For each "term" c, assign it to cover certain (i,j,k) combinations
    # Use a cyclic pattern
    
    for c in range(rank):
        # Assign this term to specific indices
        # Pattern: term c covers indices where (i+j+k) mod something == c mod something
        # Or use a more structured assignment
        
        # Simple structured pattern:
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n * m)) % p
        
        # Add some randomness to break symmetry
        if np.random.random() < 0.3:
            i_idx = np.random.randint(n)
        if np.random.random() < 0.3:
            j_idx = np.random.randint(m)
        if np.random.random() < 0.3:
            k_idx = np.random.randint(p)
        
        U[i_idx * m + j_idx, c] = 1.0
        V[j_idx * p + k_idx, c] = 1.0
        W[k_idx * n + i_idx, c] = 1.0
    
    # Add small random perturbations to help escape local minima
    U += np.random.randn(n * m, rank) * 0.1
    V += np.random.randn(m * p, rank) * 0.1
    W += np.random.randn(p * n, rank) * 0.1
    
    return U, V, W

def _random_init(rank, n, m, p, seed=0):
    """Random initialization."""
    np.random.seed(seed)
    scale = 1.0 / (rank ** 0.33)
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _alternating_minimization(rank, n, m, p, U_init=None, V_init=None, W_init=None,
                               max_iterations=2000, tol=1e-10, time_limit=None, start_time=None):
    """
    Alternating least squares to find a rank-r decomposition.
    At each step, solve linear subproblems exactly.
    """
    deadline = start_time + time_limit if time_limit else float('inf')
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    if U_init is not None:
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
    else:
        U, V, W = _random_init(rank, n, m, p, seed=np.random.randint(10000))
    
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
    
    return best_U, best_V, best_W, best_err

def _try_rank(rank, n, m, p, time_budget=10.0, start_time=None):
    """
    Try to find a decomposition of given rank using multiple strategies.
    Returns (U, V, W) as lists if found, None otherwise.
    """
    if start_time is None:
        start_time = time.time()
    
    deadline = start_time + time_budget
    best_result = None
    best_err = float('inf')
    
    # Strategy 1: Structured initialization
    for seed in range(3):
        if time.time() > deadline - 1:
            break
        U_init, V_init, W_init = _structured_init(rank, n, m, p, seed=seed)
        U_res, V_res, W_res, err = _alternating_minimization(
            rank, n, m, p,
            U_init=U_init, V_init=V_init, W_init=W_init,
            max_iterations=500,
            tol=1e-10,
            time_limit=min(3.0, max(0.5, deadline - time.time() - 1)),
            start_time=start_time
        )
        if err < best_err:
            best_err = err
            best_result = (U_res, V_res, W_res)
        if err < 1e-8:
            break
    
    # Strategy 2: Random initialization with multiple restarts
    for seed in range(5):
        if time.time() > deadline - 1:
            break
        U_init, V_init, W_init = _random_init(rank, n, m, p, seed=seed + 100)
        U_res, V_res, W_res, err = _alternating_minimization(
            rank, n, m, p,
            U_init=U_init, V_init=V_init, W_init=W_init,
            max_iterations=300,
            tol=1e-10,
            time_limit=min(2.0, max(0.3, deadline - time.time() - 1)),
            start_time=start_time
        )
        if err < best_err:
            best_err = err
            best_result = (U_res, V_res, W_res)
        if err < 1e-8:
            break
    
    # If we have a good candidate, try to snap to half-integers
    if best_result is not None and best_err < 1e-6:
        U_cand, V_cand, W_cand = best_result
        
        # Try snapping to half-integers
        U_snap = _snap_half(U_cand)
        V_snap = _snap_half(V_cand)
        W_snap = _snap_half(W_cand)
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
        
        # Try snapping to quarter-integers
        U_snap = _snap_quarter(U_cand)
        V_snap = _snap_quarter(V_cand)
        W_snap = _snap_quarter(W_cand)
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None

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
    
    # Try to find lower-rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = 30
    # Try a range of ranks, spending more time on lower ranks
    ranks_to_try = [30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 88, 89]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            break
        
        # Allocate time based on rank - lower ranks get more time
        if rank <= 33:
            time_alloc = 12.0
        elif rank <= 36:
            time_alloc = 10.0
        elif rank <= 40:
            time_alloc = 8.0
        elif rank <= 50:
            time_alloc = 6.0
        else:
            time_alloc = 4.0
        
        time_alloc = min(time_alloc, max(1.0, deadline - time.time() - 5))
        
        result = _try_rank(rank, n, m, p, time_budget=time_alloc, start_time=start_time)
        
        if result is not None:
            U_res, V_res, W_res = result
            best_U, best_V, best_W = U_res, V_res, W_res
            best_rank = rank
            # If we found a good rank, we can stop early
            if rank <= 32:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
