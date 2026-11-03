import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_decomposition_error(U, V, W, T):
    """
    Compute the error between the tensor product sum and T.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    
    error = np.linalg.norm(reconstructed - T)
    return error

def _local_search(U, V, W, T, values, max_iter=10000, timeout=10):
    """
    Local search to minimize error using Best Improvement strategy.
    U, V, W are numpy arrays.
    values is the set of allowed values.
    """
    start_time = time.time()
    current_error = _compute_decomposition_error(U, V, W, T)
    
    # Flatten indices for U, V, W
    n_u, n_v, n_w = U.size, V.size, W.size
    total_vars = n_u + n_v + n_w
    
    improved = True
    while improved and (time.time() - start_time < timeout):
        improved = False
        
        for var_idx in range(total_vars):
            if time.time() - start_time > timeout:
                break
                
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
            best_val = old_val
            best_error = current_error
            
            for new_val in values:
                if new_val == old_val:
                    continue
                    
                mat.flat[idx] = new_val
                new_error = _compute_decomposition_error(U, V, W, T)
                
                if new_error < best_error:
                    best_error = new_error
                    best_val = new_val
            
            # Revert to best_val (which might be old_val if no improvement)
            mat.flat[idx] = best_val
            
            if best_val != old_val:
                improved = True
                current_error = best_error
                
            if current_error == 0.0:
                break
                
    return current_error

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R.
    Uses gradient descent on real values followed by rounding and local search.
    """
    start_time = time.time()
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # We will try a few random restarts with gradient descent
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Random Real Initialization
        U_real = np.random.randn(n * m, R) * 0.5
        V_real = np.random.randn(m * p, R) * 0.5
        W_real = np.random.randn(p * n, R) * 0.5
        
        # Flatten for optimization
        U_flat = U_real.flatten()
        V_flat = V_real.flatten()
        W_flat = W_real.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        # Objective function
        def objective(x):
            U_r = x[:U_flat.size].reshape(n * m, R)
            V_r = x[U_flat.size:U_flat.size + V_flat.size].reshape(m * p, R)
            W_r = x[U_flat.size + V_flat.size:].reshape(p * n, R)
            
            # Compute error
            # Using einsum is efficient
            try:
                recon = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            except Exception:
                return 1e10
            err = np.sum((recon - T) ** 2)
            return err
        
        # Gradient function (optional, but helps L-BFGS)
        # We can let L-BFGS estimate it, or provide it.
        # For simplicity and speed, let's let it estimate initially, 
        # but for tensor products, gradients are easy.
        # d/dU_{ur} sum_{v,w} (sum_{r'} U_{ur'} V_{vr'} W_{wr'} - T_{uvw})^2
        # = 2 * sum_{v,w} (error_{uvw}) * V_{vr} * W_{wr}
        
        def gradient(x):
            U_r = x[:U_flat.size].reshape(n * m, R)
            V_r = x[U_flat.size:U_flat.size + V_flat.size].reshape(m * p, R)
            W_r = x[U_flat.size + V_flat.size:].reshape(p * n, R)
            
            recon = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            err_tensor = recon - T
            
            # Gradients
            # dE/dU_{ur} = 2 * sum_{v,w} err_{uvw} V_{vr} W_{wr}
            grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', err_tensor, V_r, W_r)
            grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', err_tensor, U_r, W_r)
            grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', err_tensor, U_r, V_r)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        # Run optimization
        # Limit iterations and time
        opt_res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 100, 'disp': False})
        
        if time.time() - start_time > time_limit:
            break
            
        # Extract optimized real values
        x_opt = opt_res.x
        U_opt = x_opt[:U_flat.size].reshape(n * m, R)
        V_opt = x_opt[U_flat.size:U_flat.size + V_flat.size].reshape(m * p, R)
        W_opt = x_opt[U_flat.size + V_flat.size:].reshape(p * n, R)
        
        # 2. Round to nearest half-integer
        # Multiply by 2, round, divide by 2
        U_int = np.round(U_opt * 2) / 2.0
        V_int = np.round(V_opt * 2) / 2.0
        W_int = np.round(W_opt * 2) / 2.0
        
        # 3. Verify and Local Search
        err = _compute_decomposition_error(U_int, V_int, W_int, T)
        
        # If not exact, try local search from this point
        if err != 0.0:
            # We need to ensure values are in the allowed set for local search
            # The local search uses 'values'. We should make sure U_int etc. are in 'values'
            # or just allow local search to pick from 'values'.
            # The current _local_search picks random values from 'values'.
            # So we just pass U_int as starting point.
            
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 1.0:
                err = _local_search(U_int, V_int, W_int, T, values, max_iter=5000, timeout=remaining_time)
                
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U_int.copy(), V_int.copy(), W_int.copy()
            
        if best_error == 0.0:
            break
            
    if best_error == 0.0:
        return best_U, best_V, best_W
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
    
    # Define search space
    # Integers and half-integers
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Try ranks lower than schoolbook
    # Known upper bound for 2x3x4 is likely around 18-20.
    # Let's try from 15 up to 23.
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    for R in range(15, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        # Time budget for this rank: remaining time / number of ranks left
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        # Cap time per rank to avoid spending too long on one
        time_budget = min(time_budget, 20.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify again to be safe
            err = _compute_decomposition_error(U, V, W, T)
            if err == 0.0:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                # Found a valid decomposition. 
                # Can we find an even lower one?
                # Yes, continue loop to lower R? 
                # The loop is increasing R. 
                # So if we found R=18, we should check 17, 16...
                # But the loop starts at 15. 
                # So if we find 18, we have already checked 15, 16, 17.
                # So we can break? 
                # No, we want the MINIMAL rank.
                # The loop goes 15, 16, 17...
                # So the first one found is the minimal one in this range.
                # We can break.
                break
                
    # If no better found, return schoolbook
    return best_result
