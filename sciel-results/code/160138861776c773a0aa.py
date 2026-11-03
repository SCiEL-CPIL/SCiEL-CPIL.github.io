"""Matrix multiplication 3x3x7 rank-minimal decomposition search."""
import numpy as np
import time
import random
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[a, b, c] = sum_d U[a, d] V[b, d] W[c, d]
    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-6:
                    return False
    
    valid = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid[a, b, c] = True
    
    invalid_mask = ~valid
    if np.any(np.abs(T[invalid_mask]) > 1e-6):
        return False
        
    return True

def _snap_to_half_integer(x, tol=1e-5):
    """Snap values to nearest half-integer if close, else keep as is."""
    # x is a numpy array
    snapped = np.round(x * 2) / 2.0
    return snapped

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                valid_indices.append((a, b, c))
    
    valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for (a, b, c) in valid_indices:
        valid_mask[a, b, c] = True
    
    # Schoolbook baseline (rank 63)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def calculate_error(U, V, W):
        """Calculate max norm error between computed tensor and target."""
        T = np.einsum('ad,bd,cd->abc', U, V, W)
        err = np.abs(T - T_target)
        return np.max(err)

    def frobenius_error(params, U_shape, V_shape, W_shape):
        """Calculate Frobenius norm error for scipy optimizer."""
        U_flat = params[:U_shape[0]*U_shape[1]].reshape(U_shape)
        V_flat = params[U_shape[0]*U_shape[1]:U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]].reshape(V_shape)
        W_flat = params[U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]:].reshape(W_shape)
        
        T = np.einsum('ad,bd,cd->abc', U_flat, V_flat, W_flat)
        err = T - T_target
        return np.sum(err * err)

    def local_search_discrete(U_init, V_init, W_init, target_rank, time_limit):
        """
        Perform a local search to find a valid decomposition of target_rank.
        U, V, W are numpy arrays of shape (dim, target_rank).
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Ensure initial state is snapped
        U = _snap_to_half_integer(U)
        V = _snap_to_half_integer(V)
        W = _snap_to_half_integer(W)
        
        current_err = calculate_error(U, V, W)
        if current_err < 1e-6:
            return U, V, W, True
        
        start_ls = time.time()
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() - start_ls < time_limit and no_improve_count < max_no_improve:
            # Randomly pick one of the matrices and one column to perturb
            mat_idx = random.randint(0, 2)
            col_idx = random.randint(0, target_rank - 1)
            
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
            
            # Pick a random row in that matrix to perturb
            row_idx = random.randint(0, mat.shape[0] - 1)
            
            # Generate a new value: current value + random noise
            noise_options = [-1.0, -0.5, 0.5, 1.0, 0.0]
            noise = random.choice(noise_options)
            old_val = mat[row_idx, col_idx]
            new_val = old_val + noise
            
            # Snap to half-integer
            new_val_snapped = _snap_to_half_integer(np.array([new_val]))[0]
            
            if new_val_snapped == old_val:
                continue
                
            # Apply change
            mat[row_idx, col_idx] = new_val_snapped
            
            # Calculate new error
            new_err = calculate_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                no_improve_count = 0
                if current_err < 1e-6:
                    return U, V, W, True
            else:
                # Revert
                mat[row_idx, col_idx] = old_val
                no_improve_count += 1
                
        return U, V, W, False

    # Search for lower ranks
    # Known lower bound for 3x3x7 is 21. Schoolbook is 63.
    # We try ranks from 21 upwards.
    # To save time, we limit the search time per rank.
    # Ranks 21-30 are the most promising for improvement.
    
    # If time is tight, prioritize higher ranks (closer to 63) which are easier to find?
    # Actually, lower rank is harder. But we want the best score (lowest rank).
    # So we should try lower ranks first.
    
    ranks_to_try = list(range(21, 40)) # Start from 21, go up to 39
    
    # If we have plenty of time, we can try more.
    # But 150s is a hard limit.
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
        
        # Time budget for this rank
        # Optimization is expensive. 
        # We will use Nelder-Mead to find a continuous solution, snap it, then polish.
        time_per_rank = 8.0
        
        found = False
        max_restarts = 2
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with random small values to help convergence
            # Scale: values in [-0.5, 0.5]
            U_init = np.random.uniform(-0.5, 0.5, (n*m, target_rank))
            V_init = np.random.uniform(-0.5, 0.5, (m*p, target_rank))
            W_init = np.random.uniform(-0.5, 0.5, (p*n, target_rank))
            
            # Flatten for scipy
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize Frobenius error
            # Nelder-Mead is derivative-free and robust for small problems
            # We set maxiter to avoid hanging
            try:
                res = minimize(frobenius_error, params_init, args=(U_init.shape, V_init.shape, W_init.shape), method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                
                # Extract optimized parameters
                opt_params = res.x
                U_opt = opt_params[:U_init.shape[0]*U_init.shape[1]].reshape(U_init.shape)
                V_opt = opt_params[U_init.shape[0]*U_init.shape[1]:U_init.shape[0]*U_init.shape[1]+V_init.shape[0]*V_init.shape[1]].reshape(V_init.shape)
                W_opt = opt_params[U_init.shape[0]*U_init.shape[1]+V_init.shape[0]*V_init.shape[1]:].reshape(W_init.shape)
                
                # Snap to half-integers
                U_snapped = _snap_to_half_integer(U_opt)
                V_snapped = _snap_to_half_integer(V_opt)
                W_snapped = _snap_to_half_integer(W_opt)
                
                # Check if snapped solution is valid
                U_list = [[float(x) for x in row] for row in U_snapped]
                V_list = [[float(x) for x in row] for row in V_snapped]
                W_list = [[float(x) for x in row] for row in W_snapped]
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found = True
                    break
                
                # If not valid, try to polish with discrete local search
                # This helps if the continuous solution was close but snapping introduced error
                time_left_for_polish = max(0.5, 2.0 - (time.time() - (start_time + (ranks_to_try.index(target_rank) * time_per_rank))))
                U_res, V_res, W_res, success = local_search_discrete(U_snapped, V_snapped, W_snapped, target_rank, time_left_for_polish)
                
                if success:
                    U_list = [[float(x) for x in row] for row in U_res]
                    V_list = [[float(x) for x in row] for row in V_res]
                    W_list = [[float(x) for x in row] for row in W_res]
                    
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        found = True
                        break
                        
            except Exception as e:
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
