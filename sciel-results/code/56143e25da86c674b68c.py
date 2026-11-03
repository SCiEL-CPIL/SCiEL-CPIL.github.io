import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target rank search
    # We know rank 48 works. Try lower.
    # Lower bound is ceil(48 / something)?
    # Strassen suggests significant reduction.
    # Let's try from 40 down to 20.
    
    best_result = None
    best_rank = 48
    
    # Pre-compute indices for constraints
    # Constraints: for each i, j, k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Number of constraints: n*m*p = 48
    
    def error_function(params, R):
        # params is flattened U, V, W
        # U: n*m*R, V: m*p*R, W: p*n*R
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size+V_size]
        W_flat = params[U_size+V_size:U_size+V_size+W_size]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    prod = U[u_idx] * V[v_idx] * W[w_idx]
                    s = np.sum(prod)
                    err += (s - 1.0) ** 2
        return err

    def get_solution(params, R):
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size+V_size]
        W_flat = params[U_size+V_size:U_size+V_size+W_size]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        # Convert to list of lists as required
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}

    # Try ranks
    for R in range(40, 15, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on time left
        # Estimate time per minimize call
        # Small problem, should be fast.
        
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Start with small random values.
            U_init = np.random.randn(n * m, R) * 0.1
            V_init = np.random.randn(m * p, R) * 0.1
            W_init = np.random.randn(p * n, R) * 0.1
            
            # Sometimes, scaling helps.
            # If we want sum to be 1, and R is ~40, entries should be ~1/sqrt(R)?
            # Let's try a few different scales
            
            scale = 1.0 / np.sqrt(R)
            U_init = np.random.randn(n * m, R) * scale
            V_init = np.random.randn(m * p, R) * scale
            W_init = np.random.randn(p * n, R) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(error_function, params_init, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.success or res.fun < 1e-8:
                    # Verify
                    final_err = error_function(res.x, R)
                    if final_err < 1e-8:
                        sol = get_solution(res.x, R)
                        # Round to nearest integer/half-integer if possible?
                        # The problem allows integer/half-integer.
                        # However, floating point solutions are accepted if exact?
                        # "integer/half-integer entries" is in the contract.
                        # We should try to snap to nearest 0.5.
                        
                        def snap_to_half(x):
                            return np.round(x * 2) / 2.0
                        
                        U_snapped = snap_to_half(np.array(sol["U"]))
                        V_snapped = snap_to_half(np.array(sol["V"]))
                        W_snapped = snap_to_half(np.array(sol["W"]))
                        
                        # Re-check error after snapping
                        # If snapping breaks it, we might need to return floats?
                        # The contract says "integer/half-integer entries".
                        # If we can't snap exactly, we might fail validity.
                        # But usually these tensors have rational solutions.
                        
                        # Let's check the error of the snapped version
                        U_s = U_snapped.flatten()
                        V_s = V_snapped.flatten()
                        W_s = W_snapped.flatten()
                        snapped_params = np.concatenate([U_s, V_s, W_s])
                        snapped_err = error_function(snapped_params, R)
                        
                        if snapped_err < 1e-8:
                            best_result = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                            best_rank = R
                            found = True
                            break
                        else:
                            # If snapping fails, maybe the solution is not half-integer?
                            # Or maybe we should return the float solution if the evaluator accepts it?
                            # "integer/half-integer entries" is a constraint.
                            # If we can't satisfy it, we should keep searching or return a valid schoolbook?
                            # Schoolbook is integer.
                            # If we find a float solution that is not half-integer, it might be invalid.
                            # Let's try to find a half-integer solution specifically?
                            # Or just return the float solution and hope the evaluator is lenient?
                            # No, "invalid result ... scores 0".
                            # We must stick to half-integers.
                            
                            # If snapping fails, this R might not have a half-integer solution,
                            # or our optimization didn't converge to one.
                            # Try another restart.
                            pass

            except Exception:
                continue
        
        if found:
            break

    if best_result is None:
        # Fallback to schoolbook (Rank 48)
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
        best_result = {"U": U, "V": V, "W": W}

    return best_result
