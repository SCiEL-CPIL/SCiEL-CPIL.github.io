"""Combined program: Analytic gradient optimization for tensor rank minimization."""
import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    hard_limit = 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Precompute indices for tensor operations
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def compute_tensor_and_grad(params, n, m, p):
        """
        Computes the tensor T and the gradient of the squared error (T - 1)^2.
        params: flattened array [U_flat, V_flat, W_flat]
        """
        r = len(params) // 3
        U = params[0:r].reshape(n * m, r)
        V = params[r:2*r].reshape(m * p, r)
        W = params[2*r:3*r].reshape(p * n, r)
        
        # Reshape U, V, W to 3D for easier indexing
        # U_3d[i, j, c]
        U_3d = U.reshape(n, m, r)
        # V_3d[j, k, c]
        V_3d = V.reshape(m, p, r)
        # W_3d[k, i, c]
        W_3d = W.reshape(p, n, r)
        
        # Compute T[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Using einsum:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Note: W is indexed by [k, i, c] in W_3d
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Error term
        err = T - 1.0
        obj = np.sum(err * err)
        
        # Gradients
        # dObj/dU[i,j,c] = 2 * sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
        # dObj/dV[j,k,c] = 2 * sum_i err[i,j,k] * U[i,j,c] * W[k,i,c]
        # dObj/dW[k,i,c] = 2 * sum_j err[i,j,k] * U[i,j,c] * V[j,k,c]
        
        # grad_U[i,j,c]
        # sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
        # Let's reshape err to [i, j, k]
        # V is [j, k, c], W is [k, i, c]
        # We need to sum over k.
        # grad_U_3d[i,j,c] = sum_k err[i,j,k] * V_3d[j,k,c] * W_3d[k,i,c]
        # This is a bit tricky with einsum directly because of the mixed indices.
        # Let's use loops for clarity if einsum is complex, or careful einsum.
        # einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
        grad_U_3d = np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
        
        # grad_V_3d[j,k,c] = sum_i err[i,j,k] * U_3d[i,j,c] * W_3d[k,i,c]
        # einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
        grad_V_3d = np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
        
        # grad_W_3d[k,i,c] = sum_j err[i,j,k] * U_3d[i,j,c] * V_3d[j,k,c]
        # einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)
        grad_W_3d = np.einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)
        
        grad = 2.0 * np.concatenate([
            grad_U_3d.flatten(), 
            grad_V_3d.flatten(), 
            grad_W_3d.flatten()
        ])
        
        return obj, grad

    def objective_only(params, n, m, p):
        obj, _ = compute_tensor_and_grad(params, n, m, p)
        return obj

    # Schoolbook Solution
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, c] = 1
                V_school[j*p+k, c] = 1
                W_school[k*n+i, c] = 1
                c += 1
    
    best_rank = rank_schoolbook
    best_solution = (U_school, V_school, W_school)
    best_obj = float('inf')
    
    # Try to find lower ranks
    # We start from rank_schoolbook - 1 and go down
    # We limit the search to a reasonable range to fit in time
    max_rank_to_try = rank_schoolbook - 1
    min_rank_to_try = max(30, rank_schoolbook - 20) # Try top 20 ranks first
    
    for r in range(max_rank_to_try, min_rank_to_try - 1, -1):
        if time.time() - start_time > hard_limit - 5:
            break
            
        found_valid = False
        # Number of restarts decreases as we go to lower ranks to save time?
        # Or keep constant. Let's do 5 restarts.
        num_restarts = 5
        
        for trial in range(num_restarts):
            if time.time() - start_time > hard_limit - 5:
                break
                
            random.seed(trial * 1000 + r)
            
            # Initialization: Random small values + bias
            # Bias helps avoid zero solution
            U0 = np.random.randn(n*m, r) * 0.1 + 1.0
            V0 = np.random.randn(m*p, r) * 0.1 + 1.0
            W0 = np.random.randn(p*n, r) * 0.1 + 1.0
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use L-BFGS-B with analytic gradient
                res = minimize(
                    objective_only, 
                    params0, 
                    args=(n, m, p), 
                    jac=lambda x, n, m, p: compute_tensor_and_grad(x, n, m, p)[1],
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                
                params_opt = res.x
                U_opt = params_opt[0:r].reshape(n*m, r)
                V_opt = params_opt[r:2*r].reshape(m*p, r)
                W_opt = params_opt[2*r:3*r].reshape(p*n, r)
                
                obj_val = objective_only(params_opt, n, m, p)
                
                # Check if we found a very good solution
                if obj_val < 1e-6:
                    # Try to snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify snapped solution
                    # Reconstruct T with snapped values
                    U_snap_3d = U_snap.reshape(n, m, r)
                    V_snap_3d = V_snap.reshape(m, p, r)
                    W_snap_3d = W_snap.reshape(p, n, r)
                    T_snap = np.einsum('ijc,jkc,kic->ijk', U_snap_3d, V_snap_3d, W_snap_3d)
                    
                    # Check exactness (within floating point noise)
                    if np.allclose(T_snap, 1.0, atol=1e-9):
                        best_rank = r
                        best_solution = (U_snap, V_snap, W_snap)
                        found_valid = True
                        break # Move to next (lower) rank

            except Exception:
                pass
        
        if found_valid:
            continue # Try next lower rank
            
        # If no valid solution found for this rank, continue to next lower rank
        # We don't break immediately because a lower rank might be easier to find?
        # Unlikely, but possible. However, if we fail at r, we likely fail at r-1.
        # But we have time budget. Let's keep searching until time is up.

    U_res, V_res, W_res = best_solution
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(U_res), "V": to_list(V_res), "W": to_list(W_res)}
