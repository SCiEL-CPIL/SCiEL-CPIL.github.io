import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly computes matmul."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                s = np.sum(vals)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(arr, tol=1e-10):
    """Check if all entries are integers or half-integers."""
    for val in arr:
        doubled = 2.0 * val
        if abs(doubled - round(doubled)) > tol:
            return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute sum of squared errors."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                error += (np.sum(vals) - 1.0) ** 2
    return error

def _local_search(U, V, W, n, m, p, deadline, max_iters=5000, candidates=None):
    """Local search with half-integer candidates."""
    if candidates is None:
        candidates = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    nm = n * m
    mp = m * p
    pn = p * n
    r = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    current_error = _compute_error(U, V, W, n, m, p)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        if current_error < 1e-10:
            break
        
        # Try a random perturbation
        mat_choice = np.random.randint(3)
        if mat_choice == 0:
            idx = np.random.randint(nm)
            col = np.random.randint(r)
            old_val = U[idx, col]
            new_val = np.random.choice(candidates)
            if new_val == old_val:
                continue
            U[idx, col] = new_val
        elif mat_choice == 1:
            idx = np.random.randint(mp)
            col = np.random.randint(r)
            old_val = V[idx, col]
            new_val = np.random.choice(candidates)
            if new_val == old_val:
                continue
            V[idx, col] = new_val
        else:
            idx = np.random.randint(pn)
            col = np.random.randint(r)
            old_val = W[idx, col]
            new_val = np.random.choice(candidates)
            if new_val == old_val:
                continue
            W[idx, col] = new_val
        
        new_error = _compute_error(U, V, W, n, m, p)
        
        if new_error < current_error:
            current_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            # Revert
            if mat_choice == 0:
                U[idx, col] = old_val
            elif mat_choice == 1:
                V[idx, col] = old_val
            else:
                W[idx, col] = old_val
    
    return best_U, best_V, best_W, best_error

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """Try to reduce rank from current to target_rank."""
    nm = n * m
    mp = m * p
    pn = p * n
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to zero out (current_rank - target_rank) columns
    # and compensate by adjusting other columns
    cols_to_remove = current_rank - target_rank
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Pick columns to try to remove
        cols_to_try = np.random.choice(current_rank, size=cols_to_remove, replace=False)
        
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Zero out the selected columns
        for col in cols_to_try:
            U_new[:, col] = 0
            V_new[:, col] = 0
            W_new[:, col] = 0
        
        # Now try to fix the error by adjusting remaining columns
        U_new, V_new, W_new, err = _local_search(
            U_new, V_new, W_new, n, m, p, deadline, max_iters=2000
        )
        
        if err < 1e-10:
            # Verify and check half-integer
            if _verify_decomposition(U_new, V_new, W_new, n, m, p) and \
               _is_half_integer(U_new) and _is_half_integer(V_new) and _is_half_integer(W_new):
                return U_new, V_new, W_new
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = n * m * p  # 36
    
    # Try to find lower rank solutions
    # Known: for 2x3x6, rank 34 or 35 might be possible
    
    # Strategy: Try random initializations at various ranks and do local search
    candidates = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Try ranks from 34 to 35
    for target_rank in [34, 35]:
        if time.time() > deadline - 5:
            break
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(12345 + target_rank * 100 + restart)
            
            # Initialize with random half-integer values
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Sparse initialization: most entries are 0 or 1
            for t in range(target_rank):
                # Each column should have a few non-zero entries
                num_nonzero_U = np.random.randint(1, 4)
                num_nonzero_V = np.random.randint(1, 4)
                num_nonzero_W = np.random.randint(1, 4)
                
                U_idx = np.random.choice(nm, size=num_nonzero_U, replace=False)
                V_idx = np.random.choice(mp, size=num_nonzero_V, replace=False)
                W_idx = np.random.choice(pn, size=num_nonzero_W, replace=False)
                
                for idx in U_idx:
                    U[idx, t] = np.random.choice(candidates)
                for idx in V_idx:
                    V[idx, t] = np.random.choice(candidates)
                for idx in W_idx:
                    W[idx, t] = np.random.choice(candidates)
            
            # Local search
            U, V, W, err = _local_search(U, V, W, n, m, p, deadline - 5, max_iters=3000, candidates=candidates)
            
            if err < 1e-10:
                if _verify_decomposition(U, V, W, n, m, p) and \
                   _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
        
        if best_rank == target_rank:
            break
    
    # Also try starting from schoolbook and reducing
    if best_rank == 36 and time.time() < deadline - 10:
        sb = _schoolbook(n, m, p)
        U_sb = np.array(sb["U"], dtype=float)
        V_sb = np.array(sb["V"], dtype=float)
        W_sb = np.array(sb["W"], dtype=float)
        
        for target_rank in [34, 35]:
            if time.time() > deadline - 5:
                break
            
            U_new, V_new, W_new = _try_rank_reduction(U_sb, V_sb, W_sb, n, m, p, target_rank, deadline - 5)
            
            if U_new is not None:
                best_rank = target_rank
                best_result = {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
                break
    
    return best_result
