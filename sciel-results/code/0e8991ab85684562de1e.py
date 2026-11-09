import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0

def _schoolbook():
    """
    Construct the standard schoolbook rank-60 decomposition.
    Iterates over output indices (i, k) and inner dimension j.
    W follows the k*n+i convention.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Iterate over output matrix indices (i, k) and inner dimension j
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(W[0]) if W else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _compute_residual(U_arr, V_arr, W_arr, n, m, p):
    """Compute the sum of squared residuals for the decomposition."""
    res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                res += (val - 1.0)**2
    return res

def _try_reduce_rank_structured(U, V, W, n, m, p, target_rank, time_limit=5.0, seed=0):
    """
    Try to reduce rank from current to target by structured modifications.
    Uses a coordinate descent approach on the tensor structure.
    """
    start_time = time.time()
    r_current = len(W[0])
    
    if r_current <= target_rank:
        return U, V, W
    
    # Convert to arrays
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    
    best_res = _compute_residual(U_arr, V_arr, W_arr, n, m, p)
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Try to zero out columns and see if we can still satisfy the constraints
    # We'll use a greedy approach: try to set each column to zero and check
    # if the residual remains small
    
    columns_to_try = list(range(r_current))
    random.Random(seed).shuffle(columns_to_try)
    
    for col_idx in columns_to_try:
        if time.time() - start_time > time_limit:
            break
        if r_current - 1 < target_rank:
            break
            
        # Try to zero out this column
        saved_U_col = U_arr[:, col_idx].copy()
        saved_V_col = V_arr[:, col_idx].copy()
        saved_W_col = W_arr[:, col_idx].copy()
        
        U_arr[:, col_idx] = 0
        V_arr[:, col_idx] = 0
        W_arr[:, col_idx] = 0
        
        new_res = _compute_residual(U_arr, V_arr, W_arr, n, m, p)
        
        if new_res < 1e-6:
            # We can zero out this column
            r_current -= 1
            if new_res < best_res:
                best_res = new_res
                best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
            # Keep the column zeroed
        else:
            # Restore the column
            U_arr[:, col_idx] = saved_U_col
            V_arr[:, col_idx] = saved_V_col
            W_arr[:, col_idx] = saved_W_col
    
    if r_current <= target_rank and best_res < 1e-6:
        # Trim the matrices to the reduced rank
        # Find which columns are non-zero
        non_zero_cols = []
        for c in range(r_current):
            if np.any(best_U[:, c] != 0) or np.any(best_V[:, c] != 0) or np.any(best_W[:, c] != 0):
                non_zero_cols.append(c)
        
        if len(non_zero_cols) <= target_rank:
            U_reduced = best_U[:, non_zero_cols].tolist()
            V_reduced = best_V[:, non_zero_cols].tolist()
            W_reduced = best_W[:, non_zero_cols].tolist()
            return U_reduced, V_reduced, W_reduced
    
    return None

def _random_integer_decomp(r, n, m, p, seed=0):
    """Generate a random decomposition with integer entries in {-1, 0, 1}."""
    rng = np.random.RandomState(seed)
    U = rng.randint(-1, 2, size=(n*m, r))
    V = rng.randint(-1, 2, size=(m*p, r))
    W = rng.randint(-1, 2, size=(p*n, r))
    return U.tolist(), V.tolist(), W.tolist()

def _local_search_integer(U, V, W, n, m, p, max_iters=200, temp=1.0, cooling=0.95):
    """
    Simulated Annealing on the integer lattice with faster residual computation.
    """
    r = len(W[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_res = _compute_residual(U_arr, V_arr, W_arr, n, m, p)
    best_res = current_res
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    current_temp = temp
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 15:
            break
            
        # Pick a random term c
        c = np.random.randint(0, r)
        
        # Pick a random factor and entry to perturb
        factor_choice = np.random.randint(0, 3)
        if factor_choice == 0:
            idx = np.random.randint(0, n*m)
            arr = U_arr
        elif factor_choice == 1:
            idx = np.random.randint(0, m*p)
            arr = V_arr
        else:
            idx = np.random.randint(0, p*n)
            arr = W_arr
            
        old_val = arr[idx, c]
        # Perturb by -1 or 1
        delta = np.random.choice([-1, 1])
        new_val = old_val + delta
        if abs(new_val) > 2:
            continue
            
        arr[idx, c] = new_val
        
        new_res = _compute_residual(U_arr, V_arr, W_arr, n, m, p)
        
        # Acceptance criterion
        delta_res = new_res - current_res
        if delta_res < 0 or (current_temp > 0.01 and np.random.random() < np.exp(-delta_res / current_temp)):
            current_res = new_res
            if current_res < best_res:
                best_res = current_res
                best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        else:
            arr[idx, c] = old_val
            
        current_temp *= cooling
        
        if best_res < 1e-8:
            break
            
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_res

def _try_integer_search(n, m, p, target_rank, time_limit=5.0, seed=0):
    """Try to find a valid decomposition of the given rank."""
    start = time.time()
    max_attempts = 10
    
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
            
        U, V, W = _random_integer_decomp(target_rank, n, m, p, seed=seed + attempt)
        
        # Initial check
        if _verify(U, V, W, n, m, p):
            return U, V, W
        
        # Local search with reduced iterations
        U, V, W, res = _local_search_integer(U, V, W, n, m, p, max_iters=100)
        
        if res < 1e-8:
            if _verify(U, V, W, n, m, p):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Strategy 1: Try to reduce rank from schoolbook using structured approach
    U_sb, V_sb, W_sb = best["U"], best["V"], best["W"]
    
    for target_r in range(best_rank - 1, max(20, best_rank - 5), -1):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        time_remaining = DEADLINE - 30 - (time.time() - start_time)
        if time_remaining < 1.0:
            break
            
        time_per_try = min(3.0, time_remaining / 3.0)
        
        result = _try_reduce_rank_structured(U_sb, V_sb, W_sb, n, m, p, target_r, time_limit=time_per_try, seed=target_r)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = len(W[0])
            # Update the base for further reduction
            U_sb, V_sb, W_sb = U, V, W
    
    # Strategy 2: Try random integer search for lower ranks
    for r in range(best_rank - 1, max(18, best_rank - 3), -1):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        time_remaining = DEADLINE - 20 - (time.time() - start_time)
        if time_remaining < 0.5:
            break
            
        time_per_rank = min(2.0, time_remaining / 2.0)
        
        result = _try_integer_search(n, m, p, r, time_limit=time_per_rank, seed=r*17)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = r
    
    # Return the best result found
    if time.time() - start_time > DEADLINE - 5:
        return _schoolbook()
        
    return best
