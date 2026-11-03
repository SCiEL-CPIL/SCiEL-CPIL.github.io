"""Combined program: Analytic gradient optimization for tensor rank minimization."""
import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    hard_limit = 145.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63
    
    def compute_tensor_and_grad(params, n, m, p):
        """
        Computes the tensor T and the gradient of the squared error (T - 1)^2.
        params: flattened array [U_flat, V_flat, W_flat]
        """
        r = len(params) // 3
        U = params[0:r].reshape(n * m, r)
        V = params[r:2*r].reshape(m * p, r)
        W = params[2*r:3*r].reshape(p * n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        err = T - 1.0
        obj = np.sum(err * err)
        
        grad_U_3d = np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
        grad_V_3d = np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
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

    def verify_solution(U, V, W, r, n, m, p, atol=1e-9):
        """Verify if the solution exactly decomposes the tensor."""
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, 1.0, atol=atol)

    def try_snap_and_verify(U_opt, V_opt, W_opt, r, n, m, p):
        """Try multiple snapping strategies and return best valid solution or None."""
        # Strategy 1: Round to nearest half-integer
        U1 = np.round(U_opt * 2) / 2
        V1 = np.round(V_opt * 2) / 2
        W1 = np.round(W_opt * 2) / 2
        if verify_solution(U1, V1, W1, r, n, m, p):
            return (U1, V1, W1)
        
        # Strategy 2: Round to nearest integer
        U2 = np.round(U_opt)
        V2 = np.round(V_opt)
        W2 = np.round(W_opt)
        if verify_solution(U2, V2, W2, r, n, m, p):
            return (U2, V2, W2)
        
        # Strategy 3: Floor to nearest half-integer
        U3 = np.floor(U_opt * 2) / 2
        V3 = np.floor(V_opt * 2) / 2
        W3 = np.floor(W_opt * 2) / 2
        if verify_solution(U3, V3, W3, r, n, m, p):
            return (U3, V3, W3)
        
        # Strategy 4: Ceiling to nearest half-integer
        U4 = np.ceil(U_opt * 2) / 2
        V4 = np.ceil(V_opt * 2) / 2
        W4 = np.ceil(W_opt * 2) / 2
        if verify_solution(U4, V4, W4, r, n, m, p):
            return (U4, V4, W4)
        
        # Strategy 5: Mix - round most values, keep some as-is if very close
        for snap_type in ['round', 'floor', 'ceil']:
            U5 = U_opt.copy()
            V5 = V_opt.copy()
            W5 = W_opt.copy()
            # Only snap values that are close to half-integers
            mask_U = np.abs(U_opt * 2 - np.round(U_opt * 2)) < 0.1
            mask_V = np.abs(V_opt * 2 - np.round(V_opt * 2)) < 0.1
            mask_W = np.abs(W_opt * 2 - np.round(W_opt * 2)) < 0.1
            if snap_type == 'round':
                U5[mask_U] = np.round(U5[mask_U] * 2) / 2
                V5[mask_V] = np.round(V5[mask_V] * 2) / 2
                W5[mask_W] = np.round(W5[mask_W] * 2) / 2
            elif snap_type == 'floor':
                U5[mask_U] = np.floor(U5[mask_U] * 2) / 2
                V5[mask_V] = np.floor(V5[mask_V] * 2) / 2
                W5[mask_W] = np.floor(W5[mask_W] * 2) / 2
            else:
                U5[mask_U] = np.ceil(U5[mask_U] * 2) / 2
                V5[mask_V] = np.ceil(V5[mask_V] * 2) / 2
                W5[mask_W] = np.ceil(W5[mask_W] * 2) / 2
            if verify_solution(U5, V5, W5, r, n, m, p):
                return (U5, V5, W5)
        
        return None

    def polish_solution(params, n, m, p, r, max_iter=2000, ftol=1e-18, gtol=1e-14):
        """High-precision polish pass with tighter tolerances."""
        try:
            res = minimize(
                objective_only, 
                params, 
                args=(n, m, p), 
                jac=lambda x, n, m, p: compute_tensor_and_grad(x, n, m, p)[1],
                method='L-BFGS-B', 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            return res.x
        except Exception:
            return params

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
    
    # Try to find lower ranks - focus on promising range
    # Known results suggest rank around 21-25 for 3x3x7
    max_rank_to_try = rank_schoolbook - 1
    min_rank_to_try = 18  # More aggressive lower bound
    
    def generate_initializations(r, n, m, p, seed_base):
        """Generate diverse initializations for a given rank."""
        inits = []
        
        # Strategy 1: Random + bias toward 1
        random.seed(seed_base)
        U0 = np.random.randn(n*m, r) * 0.3 + 0.5
        V0 = np.random.randn(m*p, r) * 0.3 + 0.5
        W0 = np.random.randn(p*n, r) * 0.3 + 0.5
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        # Strategy 2: Pure random
        random.seed(seed_base + 1)
        U0 = np.random.randn(n*m, r) * 0.5
        V0 = np.random.randn(m*p, r) * 0.5
        W0 = np.random.randn(p*n, r) * 0.5
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        # Strategy 3: Structured - leverage matrix multiplication structure
        random.seed(seed_base + 2)
        U0 = np.zeros((n*m, r), dtype=float)
        V0 = np.zeros((m*p, r), dtype=float)
        W0 = np.zeros((p*n, r), dtype=float)
        
        # Distribute columns to cover different (i,j) and (j,k) pairs
        cols_per_group = max(1, r // (n * m))
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for c_local in range(min(cols_per_group, r - col_idx)):
                    c = col_idx + c_local
                    if c >= r:
                        break
                    U0[i*m+j, c] = 1.0
                    for k in range(p):
                        V0[j*p+k, c] = 1.0
                        W0[k*n+i, c] = 1.0
                col_idx += cols_per_group
                if col_idx >= r:
                    break
            if col_idx >= r:
                break
        
        # Add small random perturbation
        U0 += np.random.randn(n*m, r) * 0.02
        V0 += np.random.randn(m*p, r) * 0.02
        W0 += np.random.randn(p*n, r) * 0.02
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        # Strategy 4: Sparse with structured sparsity
        random.seed(seed_base + 3)
        U0 = np.zeros((n*m, r), dtype=float)
        V0 = np.zeros((m*p, r), dtype=float)
        W0 = np.zeros((p*n, r), dtype=float)
        
        for c in range(r):
            # Each column focuses on a subset of the multiplication
            i_base = c % n
            j_base = (c // n) % m
            U0[i_base*m + j_base, c] = 1.0
            # Random subset of k values
            k_subset = np.random.choice(p, size=max(1, p//2), replace=False)
            for k in k_subset:
                V0[j_base*p+k, c] = 1.0
                W0[k*n+i_base, c] = 1.0
        
        U0 += np.random.randn(n*m, r) * 0.05
        V0 += np.random.randn(m*p, r) * 0.05
        W0 += np.random.randn(p*n, r) * 0.05
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        # Strategy 5: Kronecker-like structure
        random.seed(seed_base + 4)
        U0 = np.zeros((n*m, r), dtype=float)
        V0 = np.zeros((m*p, r), dtype=float)
        W0 = np.zeros((p*n, r), dtype=float)
        
        # Try to create a structure where each column handles a "block"
        for c in range(r):
            # Assign this column to handle certain (i,j,k) combinations
            i_val = c % n
            j_val = (c // n) % m
            U0[i_val*m + j_val, c] = 1.0
            # Distribute k values cyclically
            for k in range(p):
                if (c + k) % 3 == 0:  # Sparse k assignment
                    V0[j_val*p+k, c] = 1.0
                    W0[k*n+i_val, c] = 1.0
        
        U0 += np.random.randn(n*m, r) * 0.03
        V0 += np.random.randn(m*p, r) * 0.03
        W0 += np.random.randn(p*n, r) * 0.03
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        return inits

    # Search ranks from high to low, but focus more on promising range
    # Weight restarts based on how promising the rank is
    for r in range(max_rank_to_try, min_rank_to_try - 1, -1):
        if time.time() - start_time > hard_limit - 5:
            break
            
        found_valid = False
        
        # Allocate more time to ranks near the expected minimum
        # Ranks 20-25 get more restarts
        time_remaining = hard_limit - 5 - (time.time() - start_time)
        ranks_remaining = r - min_rank_to_try + 1
        
        # Weighted time allocation
        if 19 <= r <= 25:
            time_weight = 3.0
        elif 26 <= r <= 35:
            time_weight = 1.5
        else:
            time_weight = 1.0
        
        time_per_rank = (time_remaining / max(1, ranks_remaining)) * time_weight
        est_time_per_trial = 2.5
        num_restarts = max(5, min(30, int(time_per_rank / est_time_per_trial)))
        
        for trial in range(num_restarts):
            if time.time() - start_time > hard_limit - 5:
                break
                
            inits = generate_initializations(r, n, m, p, trial * 1000 + r)
            params0 = inits[trial % len(inits)]
            
            try:
                # Use L-BFGS-B with analytic gradient
                res = minimize(
                    objective_only, 
                    params0, 
                    args=(n, m, p), 
                    jac=lambda x, n, m, p: compute_tensor_and_grad(x, n, m, p)[1],
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )
                
                params_opt = res.x
                U_opt = params_opt[0:r].reshape(n*m, r)
                V_opt = params_opt[r:2*r].reshape(m*p, r)
                W_opt = params_opt[2*r:3*r].reshape(p*n, r)
                
                obj_val = objective_only(params_opt, n, m, p)
                
                # Check if we found a very good solution
                if obj_val < 1e-3:
                    # Polish with tighter tolerances
                    if time.time() - start_time < hard_limit - 10:
                        params_polished = polish_solution(params_opt, n, m, p, r)
                        U_opt = params_polished[0:r].reshape(n*m, r)
                        V_opt = params_polished[r:2*r].reshape(m*p, r)
                        W_opt = params_polished[2*r:3*r].reshape(p*n, r)
                        obj_val = objective_only(params_polished, n, m, p)
                    
                    # Try multiple snapping strategies
                    snapped = try_snap_and_verify(U_opt, V_opt, W_opt, r, n, m, p)
                    if snapped is not None:
                        U_snap, V_snap, W_snap = snapped
                        best_rank = r
                        best_solution = (U_snap, V_snap, W_snap)
                        found_valid = True
                        break  # Move to next (lower) rank

            except Exception:
                pass
        
        if found_valid:
            continue  # Try next lower rank

    U_res, V_res, W_res = best_solution
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(U_res), "V": to_list(V_res), "W": to_list(W_res)}
