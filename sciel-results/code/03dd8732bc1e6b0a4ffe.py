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
    rank_schoolbook = n * m * p
    
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

    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integers."""
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

    def polish_solution(params, n, m, p, r, max_iter=1000, ftol=1e-16, gtol=1e-12):
        """High-precision polish pass with tighter tolerances."""
        try:
            res = minimize(
                objective_only, 
                params, 
                args=(n, m, p), 
                jac=lambda x, n, m, p: compute_tensor_and_grad(x, n, m, p)[1],
                method='L-BFGS-B', 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
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
    
    # Try to find lower ranks
    max_rank_to_try = rank_schoolbook - 1
    min_rank_to_try = max(25, rank_schoolbook - 40)  # Extended search range
    
    def generate_initializations(r, n, m, p, seed_base):
        """Generate diverse initializations for a given rank."""
        inits = []
        
        # Strategy 1: Random + bias
        random.seed(seed_base)
        U0 = np.random.randn(n*m, r) * 0.1 + 1.0
        V0 = np.random.randn(m*p, r) * 0.1 + 1.0
        W0 = np.random.randn(p*n, r) * 0.1 + 1.0
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        # Strategy 2: Larger random initialization
        random.seed(seed_base + 1)
        U0 = np.random.randn(n*m, r) * 0.5
        V0 = np.random.randn(m*p, r) * 0.5
        W0 = np.random.randn(p*n, r) * 0.5
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        # Strategy 3: Structured initialization based on schoolbook pattern
        if r >= n * m:
            random.seed(seed_base + 2)
            U0 = np.zeros((n*m, r), dtype=float)
            V0 = np.zeros((m*p, r), dtype=float)
            W0 = np.zeros((p*n, r), dtype=float)
            
            # Use first n*m columns from schoolbook-like pattern
            for c_idx in range(min(r, n*m)):
                i = c_idx % n
                j = (c_idx // n) % m
                U0[i*m+j, c_idx] = 1.0
                for k in range(p):
                    V0[j*p+k, c_idx] = 1.0
                    W0[k*n+i, c_idx] = 1.0
            
            # Add random perturbation
            U0 += np.random.randn(n*m, r) * 0.05
            V0 += np.random.randn(m*p, r) * 0.05
            W0 += np.random.randn(p*n, r) * 0.05
            inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        # Strategy 4: Sparse random initialization
        random.seed(seed_base + 3)
        U0 = np.zeros((n*m, r), dtype=float)
        V0 = np.zeros((m*p, r), dtype=float)
        W0 = np.zeros((p*n, r), dtype=float)
        
        for c in range(r):
            # Randomly select non-zero entries
            nz_U = np.random.choice(n*m, size=min(3, n*m), replace=False)
            nz_V = np.random.choice(m*p, size=min(3, m*p), replace=False)
            nz_W = np.random.choice(p*n, size=min(3, p*n), replace=False)
            
            U0[nz_U, c] = np.random.uniform(0.5, 1.5, size=len(nz_U))
            V0[nz_V, c] = np.random.uniform(0.5, 1.5, size=len(nz_V))
            W0[nz_W, c] = np.random.uniform(0.5, 1.5, size=len(nz_W))
        
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
        return inits

    for r in range(max_rank_to_try, min_rank_to_try - 1, -1):
        if time.time() - start_time > hard_limit - 10:
            break
            
        found_valid = False
        # Adaptive number of restarts based on remaining time
        time_remaining = hard_limit - 10 - (time.time() - start_time)
        ranks_remaining = r - min_rank_to_try + 1
        time_per_rank = time_remaining / max(1, ranks_remaining)
        
        # Estimate time per optimization (~2-5s per trial)
        est_time_per_trial = 3.0
        num_restarts = max(3, min(20, int(time_per_rank / est_time_per_trial)))
        
        for trial in range(num_restarts):
            if time.time() - start_time > hard_limit - 10:
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
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                params_opt = res.x
                U_opt = params_opt[0:r].reshape(n*m, r)
                V_opt = params_opt[r:2*r].reshape(m*p, r)
                W_opt = params_opt[2*r:3*r].reshape(p*n, r)
                
                obj_val = objective_only(params_opt, n, m, p)
                
                # Check if we found a very good solution
                if obj_val < 1e-4:
                    # Polish with tighter tolerances
                    if time.time() - start_time < hard_limit - 15:
                        params_polished = polish_solution(params_opt, n, m, p, r)
                        U_opt = params_polished[0:r].reshape(n*m, r)
                        V_opt = params_polished[r:2*r].reshape(m*p, r)
                        W_opt = params_polished[2*r:3*r].reshape(p*n, r)
                        obj_val = objective_only(params_polished, n, m, p)
                    
                    # Try to snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    
                    # Verify snapped solution
                    if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                        best_rank = r
                        best_solution = (U_snap, V_snap, W_snap)
                        found_valid = True
                        break  # Move to next (lower) rank
                    else:
                        # Try integer snapping as fallback
                        U_int = np.round(U_opt)
                        V_int = np.round(V_opt)
                        W_int = np.round(W_opt)
                        if verify_solution(U_int, V_int, W_int, r, n, m, p):
                            best_rank = r
                            best_solution = (U_int, V_int, W_int)
                            found_valid = True
                            break

            except Exception:
                pass
        
        if found_valid:
            continue  # Try next lower rank

    U_res, V_res, W_res = best_solution
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(U_res), "V": to_list(V_res), "W": to_list(W_res)}
