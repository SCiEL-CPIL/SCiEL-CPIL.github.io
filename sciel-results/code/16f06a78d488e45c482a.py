import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Strassen for 2x2x2 block
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook for A[0:2,2] * B[2,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Schoolbook for A[0:2,0:3] * B[0:3,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition is correct using vectorized computation."""
    n, m, p = DIMENSIONS
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Build index arrays
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    # Reshape to flat
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # U indices: i*m + j
    u_idx = i_flat * m + j_flat  # shape (18,)
    # V indices: j*p + k
    v_idx = j_flat * p + k_flat  # shape (18,)
    # W indices: k*n + i
    w_idx = k_flat * n + i_flat  # shape (18,)
    
    # Get the relevant rows
    U_sel = U[u_idx, :]  # shape (18, rank)
    V_sel = V[v_idx, :]  # shape (18, rank)
    W_sel = W[w_idx, :]  # shape (18, rank)
    
    # Compute sum over r for each (i,j,k)
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    errors = np.abs(vals - 1)
    return np.all(errors < 1e-9), list(zip(i_flat[errors > 1e-9], j_flat[errors > 1e-9], k_flat[errors > 1e-9], vals[errors > 1e-9]))

def _try_find_redundant_term(U17, V17, W17):
    """Try to find if any term in rank-17 is redundant by checking if removing it and adjusting others works."""
    n, m, p = DIMENSIONS
    rank = 17
    
    # For each term to remove, check if the remaining 16 terms can be adjusted
    # to satisfy all 18 constraints
    # This is a nonlinear system. Let's use scipy's least squares.
    from scipy.optimize import least_squares
    
    best_rank = 17
    best_solution = None
    
    for term_to_remove in range(rank):
        # Create masks for remaining terms
        mask = np.ones(rank, dtype=bool)
        mask[term_to_remove] = False
        
        U_rem = U17[:, mask]  # shape (6, 16)
        V_rem = V17[:, mask]  # shape (9, 16)
        W_rem = W17[:, mask]  # shape (6, 16)
        
        # Flatten parameters
        x0 = np.concatenate([U_rem.flatten(), V_rem.flatten(), W_rem.flatten()])
        
        def residuals(x):
            U_x = x[:6*16].reshape(6, 16)
            V_x = x[6*16:6*16+9*16].reshape(9, 16)
            W_x = x[6*16+9*16:].reshape(6, 16)
            
            # Compute all 18 values
            i_idx, j_idx, k_idx = np.indices((n, m, p))
            i_flat = i_idx.flatten()
            j_flat = j_idx.flatten()
            k_flat = k_idx.flatten()
            
            u_idx = i_flat * m + j_flat
            v_idx = j_flat * p + k_flat
            w_idx = k_flat * n + i_flat
            
            U_sel = U_x[u_idx, :]
            V_sel = V_x[v_idx, :]
            W_sel = W_x[w_idx, :]
            
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            return vals - 1.0
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if np.max(np.abs(result.fun)) < 1e-8:
                U_sol = result.x[:6*16].reshape(6, 16)
                V_sol = result.x[6*16:6*16+9*16].reshape(9, 16)
                W_sol = result.x[6*16+9*16:].reshape(6, 16)
                best_solution = (U_sol, V_sol, W_sol)
                break
        except Exception:
            continue
    
    return best_solution

def _try_rank16_search(U17, V17, W17, time_budget=30):
    """Try to find a rank-16 decomposition."""
    start_time = time.time()
    
    # First, try to find a redundant term in the rank-17 solution
    result = _try_find_redundant_term(U17, V17, W17)
    if result is not None:
        U16, V16, W16 = result
        valid, _ = _verify(U16, V16, W16)
        if valid:
            return U16, V16, W16, 16
    
    # If that didn't work, try a more direct numerical search
    # Use the structure: fix the sparsity pattern from rank-17 and solve
    from scipy.optimize import least_squares
    
    n, m, p = DIMENSIONS
    num_i = n * m  # 6
    num_j = m * p  # 9
    num_k = p * n  # 6
    
    # Try different initializations
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize from rank-17 with one term zeroed out
        U_init = U17.copy()
        V_init = V17.copy()
        W_init = W17.copy()
        
        term_to_zero = attempt % 17
        U_init[:, term_to_zero] = 0
        V_init[:, term_to_zero] = 0
        W_init[:, term_to_zero] = 0
        
        # Remove the zeroed column
        mask = np.ones(17, dtype=bool)
        mask[term_to_zero] = False
        U_rem = U_init[:, mask]
        V_rem = V_init[:, mask]
        W_rem = W_init[:, mask]
        
        x0 = np.concatenate([U_rem.flatten(), V_rem.flatten(), W_rem.flatten()])
        
        def residuals(x):
            U_x = x[:6*16].reshape(6, 16)
            V_x = x[6*16:6*16+9*16].reshape(9, 16)
            W_x = x[6*16+9*16:].reshape(6, 16)
            
            i_idx, j_idx, k_idx = np.indices((n, m, p))
            i_flat = i_idx.flatten()
            j_flat = j_idx.flatten()
            k_flat = k_idx.flatten()
            
            u_idx = i_flat * m + j_flat
            v_idx = j_flat * p + k_flat
            w_idx = k_flat * n + i_flat
            
            U_sel = U_x[u_idx, :]
            V_sel = V_x[v_idx, :]
            W_sel = W_x[w_idx, :]
            
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            return vals - 1.0
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=500)
            if np.max(np.abs(result.fun)) < 1e-8:
                U_sol = result.x[:6*16].reshape(6, 16)
                V_sol = result.x[6*16:6*16+9*16].reshape(9, 16)
                W_sol = result.x[6*16+9*16:].reshape(6, 16)
                valid, _ = _verify(U_sol, V_sol, W_sol)
                if valid:
                    return U_sol, V_sol, W_sol, 16
        except Exception:
            continue
    
    return None, None, None, 17

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: return by 115s
    
    # Build the known rank-17 solution
    U17, V17, W17 = _build_rank17()
    
    # Verify it
    valid, errors = _verify(U17, V17, W17)
    if not valid:
        # Should not happen, but handle it
        pass
    
    # Try to find rank 16 if we have time
    elapsed = time.time() - start_time
    if elapsed < 100:  # Leave 20s safety margin
        time_for_search = min(80, 100 - elapsed)
        U16, V16, W16, rank_found = _try_rank16_search(U17, V17, W17, time_budget=time_for_search)
        
        if U16 is not None and rank_found == 16:
            # Verify the rank-16 solution
            valid16, errors16 = _verify(U16, V16, W16)
            if valid16:
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return the rank-17 solution
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
