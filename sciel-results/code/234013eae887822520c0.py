import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """Build target tensor T of shape (n*m, m*p, p*n)"""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _get_schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _compute_error_fast(U, V, W, T):
    """Compute ||U V^T W^T - T||_F efficiently using einsum."""
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    diff = reconstructed - T
    return np.sqrt(np.sum(diff * diff))

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    # Build target
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    diff = reconstructed - T
    return np.max(np.abs(diff)) < tol

def _local_search(U, V, W, T, R, time_limit, start_time, values=None):
    """Local search to minimize error. Returns (U, V, W, error)."""
    if values is None:
        values = [-2, -1, 0, 1, 2]
    
    current_error = _compute_error_fast(U, V, W, T)
    n, m, p = T.shape[0] // 3, 3, T.shape[2] // 2  # Infer from context
    # Actually let's pass dimensions
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    
    max_iter = 5000
    stall_count = 0
    stall_limit = 200
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if current_error < 1e-12:
            break
        
        # Pick a random entry to modify
        total = U.size + V.size + W.size
        var_idx = random.randint(0, total - 1)
        
        if var_idx < U.size:
            mat = U
            local_idx = var_idx
        elif var_idx < U.size + V.size:
            mat = V
            local_idx = var_idx - U.size
        else:
            mat = W
            local_idx = var_idx - U.size - V.size
        
        old_val = mat.flat[local_idx]
        new_val = random.choice(values)
        
        mat.flat[local_idx] = new_val
        new_error = _compute_error_fast(U, V, W, T)
        
        if new_error < current_error - 1e-15:
            current_error = new_error
            stall_count = 0
        else:
            mat.flat[local_idx] = old_val
            stall_count += 1
        
        if stall_count > stall_limit:
            break
    
    return U, V, W, current_error

def _try_rank(R, n, m, p, T, time_limit, start_time):
    """Try to find rank-R decomposition. Returns (U,V,W) or None."""
    nm, mp, pn = n * m, m * p, p * n
    values = [-1, 0, 1]
    
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    num_restarts = 30
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize
        U = np.random.choice(values, size=(nm, R)).astype(np.float64)
        V = np.random.choice(values, size=(mp, R)).astype(np.float64)
        W = np.random.choice(values, size=(pn, R)).astype(np.float64)
        
        # Set some entries to ensure coverage
        # Try to make each row of U, V, W have at least one non-zero
        for r in range(R):
            if np.all(U[:, r] == 0):
                U[random.randint(0, nm - 1), r] = 1.0
            if np.all(V[:, r] == 0):
                V[random.randint(0, mp - 1), r] = 1.0
            if np.all(W[:, r] == 0):
                W[random.randint(0, pn - 1), r] = 1.0
        
        U, V, W, err = _local_search(U, V, W, T, R, time_limit, start_time, values)
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Lower bound: max(nm, mp, pn) = max(6, 12, 8) = 12
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower bound up
    for R in range(lower_bound, best_rank):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining = deadline - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time based on remaining ranks
        ranks_left = best_rank - R
        time_budget = min(15.0, remaining / max(ranks_left, 1))
        if time_budget < 1.5:
            break
        
        result = _try_rank(R, n, m, p, T, time_budget, start_time)
        
        if result is not None:
            U, V, W = result
            if _check_exact(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
