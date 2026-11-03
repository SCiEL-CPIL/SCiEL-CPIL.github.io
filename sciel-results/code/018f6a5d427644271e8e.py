import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T

def _compute_error_int(U, V, W, T):
    """
    Compute the sum of squared errors using integer arithmetic.
    U, V, W are integer arrays (scaled by 2).
    T is integer array (target values, 1 or 0).
    The reconstruction is sum_r U[:,r] V[:,r] W[:,r].
    If U,V,W are scaled by 2, the result is scaled by 8.
    So we compare (recon/8) vs T, or equivalently: recon_int vs 8*T.
    """
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    target_scaled = 8 * T
    diff = recon - target_scaled
    return int(np.sum(diff * diff))

def _local_search_int(U, V, W, T, values, max_iter=5000, timeout=5.0):
    """
    Discrete local search using integer arithmetic.
    U, V, W are integer arrays.
    values are the allowed integer values (scaled by 2).
    """
    start_time = time.time()
    current_error = _compute_error_int(U, V, W, T)
    
    n_u, n_v, n_w = U.size, V.size, W.size
    total_vars = n_u + n_v + n_w
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error == 0:
            break
            
        # Pick a random variable
        var_idx = random.randint(0, total_vars - 1)
        
        if var_idx < n_u:
            mat = U
            idx = var_idx
        elif var_idx < n_u + n_v:
            mat = V
            idx = var_idx - n_u
        else:
            mat = W
            idx = var_idx - n_u - n_v
            
        old_val = mat.flat[idx]
        
        # Try all possible new values (best improvement)
        best_new_val = old_val
        best_new_error = current_error
        
        for new_val in values:
            if new_val == old_val:
                continue
            mat.flat[idx] = new_val
            new_error = _compute_error_int(U, V, W, T)
            if new_error < best_new_error:
                best_new_error = new_error
                best_new_val = new_val
        
        # Apply the best change
        if best_new_val != old_val:
            mat.flat[idx] = best_new_val
            current_error = best_new_error
        else:
            mat.flat[idx] = old_val

    return current_error

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R using discrete search.
    """
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    int_values = values  # Already scaled integers
    
    # Multiple restarts
    max_restarts = 30
    for i in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Sparse initialization: mostly zeros with some random non-zeros
        density = 0.3
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        
        for r in range(R):
            # Each column has some non-zero entries
            num_nonzero_u = max(1, int(n * m * density))
            num_nonzero_v = max(1, int(m * p * density))
            num_nonzero_w = max(1, int(p * n * density))
            
            u_indices = random.sample(range(n * m), num_nonzero_u)
            v_indices = random.sample(range(m * p), num_nonzero_v)
            w_indices = random.sample(range(p * n), num_nonzero_w)
            
            for idx in u_indices:
                U[idx, r] = random.choice(int_values)
            for idx in v_indices:
                V[idx, r] = random.choice(int_values)
            for idx in w_indices:
                W[idx, r] = random.choice(int_values)
        
        # Run local search
        time_per_restart = (time_limit - (time.time() - start_time)) / (max_restarts - i)
        err = _local_search_int(U, V, W, T, int_values, max_iter=3000, timeout=time_per_restart)
        
        if err < best_error:
            best_error = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error == 0:
            break
            
    if best_error == 0:
        # Convert back to floats (divide by 2)
        return best_U / 2.0, best_V / 2.0, best_W / 2.0
    return None

def _try_rank_from_schoolbook(R, n, m, p, T, values, time_limit):
    """
    Try to reduce rank from schoolbook by merging columns.
    Start with schoolbook and try to zero out columns while maintaining exactness.
    """
    start_time = time.time()
    
    # Build schoolbook
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb), dtype=int)
    V = np.zeros((m * p, rank_sb), dtype=int)
    W = np.zeros((p * n, rank_sb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 2  # Scaled by 2
                V[j * p + k, col] = 2
                W[k * n + i, col] = 2
                col += 1
    
    # We want to reduce to rank R. Try to set (rank_sb - R) columns to zero
    # and adjust others to compensate.
    # This is hard, so let's just do random restarts with target rank R
    
    int_values = values
    
    max_restarts = 20
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for i in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize near schoolbook structure but with rank R
        # Use a subset of the schoolbook patterns
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        
        # Assign each schoolbook term to one of R slots
        assignments = [random.randint(0, R - 1) for _ in range(rank_sb)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    slot = assignments[col_idx]
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    # Accumulate (may need to adjust for collisions)
                    U[u, slot] += 2
                    V[v, slot] += 2
                    W[w, slot] += 2
                    col_idx += 1
        
        # Now U, V, W may have values > 2, clip to allowed range
        for arr in [U, V, W]:
            arr[arr > max(values)] = max(values)
            arr[arr < min(values)] = min(values)
        
        time_per_restart = (time_limit - (time.time() - start_time)) / (max_restarts - i)
        err = _local_search_int(U, V, W, T, int_values, max_iter=2000, timeout=time_per_restart)
        
        if err < best_error:
            best_error = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error == 0:
            break
            
    if best_error == 0:
        return best_U / 2.0, best_V / 2.0, best_W / 2.0
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Define search space: half-integers scaled by 2
    # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    # Scaled: -4, -3, -2, -1, 0, 1, 2, 3, 4
    values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Try ranks from high to low, spending more time on likely-successful ranks
    # Based on literature, rank 18-20 is likely achievable
    ranks_to_try = [20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
        if time.time() - start_time > deadline:
            break
            
        # Allocate time based on rank: higher ranks get more time
        # Ranks 18-20 get 20s, 15-17 get 15s, lower get 10s
        if R >= 18:
            time_budget = 20.0
        elif R >= 15:
            time_budget = 15.0
        else:
            time_budget = 10.0
            
        if time.time() - start_time + time_budget > deadline:
            time_budget = deadline - (time.time() - start_time) - 5.0
        if time_budget < 2.0:
            break
            
        # Try both random and schoolbook-based initialization
        result = _try_rank(R, n, m, p, T, values, time_budget * 0.5)
        
        if result is None and time.time() - start_time < deadline - 5.0:
            result = _try_rank_from_schoolbook(R, n, m, p, T, values, time_budget * 0.5)
        
        if result is not None:
            U, V, W = result
            # Verify
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            if np.allclose(recon, T, atol=1e-9):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
