"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 150s limit, use 140s as hard stop with 10s margin for safety
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Target tensor T
    # T[u, v, w] = 1 if u%3 == v//3 and v%3 == w//3 and w%3 == u//3
    target_flat = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_flat[u, v, w] = 1.0
    target_flat_list = target_flat.flatten()
    
    # Precompute for fast tensor computation
    def compute_tensor(U, V, W):
        """Fast vectorized tensor computation: T = sum_r U_r * V_r * W_r"""
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def error_func(params, rank):
        # Check deadline
        if time.time() > deadline:
            return 1e10
            
        U = params[:rank * n * m].reshape((n * m, rank))
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
        W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
        
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_flat_list
        return np.dot(diff, diff)
    
    def compute_gradient(params, rank):
        U = params[:rank * n * m].reshape((n * m, rank))
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
        W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
        
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_flat_list
        diff_3d = diff.reshape((n * m, m * p, p * n))
        
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff_3d, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff_3d, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff_3d, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def verify_and_round(U, V, W):
        """Try multiple rounding strategies and verify"""
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_flat_list
        err = np.dot(diff, diff)
        
        if err > 1e-6:
            return None, None, None, False
        
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        T_rounded = compute_tensor(U_r, V_r, W_r)
        diff_r = T_rounded.flatten() - target_flat_list
        err_r = np.dot(diff_r, diff_r)
        
        if err_r < 1e-6:
            return U_r.tolist(), V_r.tolist(), W_r.tolist(), True
        
        # Try rounding to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        
        T_int = compute_tensor(U_i, V_i, W_i)
        diff_i = T_int.flatten() - target_flat_list
        err_i = np.dot(diff_i, diff_i)
        
        if err_i < 1e-6:
            return U_i.tolist(), V_i.tolist(), W_i.tolist(), True
        
        return None, None, None, False

    # Schoolbook solution (Rank 27)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

    # Verify schoolbook solution works
    T_s = compute_tensor(U_school, V_school, W_school)
    diff_s = T_s.flatten() - target_flat_list
    err_s = np.dot(diff_s, diff_s)
    if err_s > 1e-10:
        # Fallback
        return best_result

    # Try to find lower ranks (23 is the target)
    target_ranks = [23, 24, 25, 26]
    
    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Time budget per rank
        rank_budget = 40.0 if rank == 23 else 20.0
        rank_start = time.time()
        
        # Strategy 1: Use known rank-23 structure as initialization
        # Based on the idea that rank-23 decompositions exist for 3x3 matrix multiplication
        
        num_trials = 0
        while time.time() - rank_start < rank_budget:
            if time.time() > deadline - 5:
                break
                
            num_trials += 1
            np.random.seed(num_trials * 123 + rank)
            
            # Strategy A: Dense random initialization with structured perturbation
            U_init = np.random.randn(n * m, rank) * 0.3
            V_init = np.random.randn(m * p, rank) * 0.3
            W_init = np.random.randn(p * n, rank) * 0.3
            
            # Add structured components that resemble matrix multiplication patterns
            for r in range(rank):
                if r < 9:
                    # Add diagonal-like patterns
                    for i in range(n):
                        U_init[i * m + i, r] += 1.0
                        V_init[i * p + i, r] += 1.0
                        W_init[i * n + i, r] += 1.0
                elif r < 18:
                    # Add off-diagonal patterns
                    idx = (r - 9) % 9
                    i1, i2 = idx // 3, idx % 3
                    if i1 != i2:
                        U_init[i1 * m + i2, r] += 1.0
                        V_init[i2 * p + i1, r] += 1.0
                        W_init[i1 * n + i2, r] += 1.0
                else:
                    # Add mixed patterns
                    idx = (r - 18) % 5
                    i1, i2 = idx // 3, idx % 3
                    U_init[i1 * m + i2, r] += 0.5
                    V_init[i2 * p + i1, r] += 0.5
                    W_init[i1 * n + i2, r] += 0.5
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(error_func, params, args=(rank,), 
                             jac=lambda p, r: compute_gradient(p, r),
                             method='L-BFGS-B', 
                             options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < 1e-8:
                    U_final = res.x[:rank * n * m].reshape((n * m, rank))
                    V_final = res.x[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
                    W_final = res.x[rank * n * m + rank * m * p :].reshape((p * n, rank))
                    
                    U_r, V_r, W_r, ok = verify_and_round(U_final, V_final, W_final)
                    if ok:
                        best_result = {
                            "U": U_r,
                            "V": V_r,
                            "W": W_r
                        }
                        return best_result
            except Exception:
                pass

            # Strategy B: Subset Initialization from Schoolbook (every 2nd trial)
            if num_trials % 2 == 0:
                # Select a subset of schoolbook columns
                indices = np.random.choice(rank_school, rank, replace=False)
                
                U_init2 = np.zeros((n * m, rank))
                V_init2 = np.zeros((m * p, rank))
                W_init2 = np.zeros((p * n, rank))
                
                for new_idx, old_idx in enumerate(indices):
                    U_init2[:, new_idx] = U_school[:, old_idx]
                    V_init2[:, new_idx] = V_school[:, old_idx]
                    W_init2[:, new_idx] = W_school[:, old_idx]
                
                # Add small noise to break symmetry and allow optimization to move
                U_init2 += np.random.normal(0, 0.1, U_init2.shape)
                V_init2 += np.random.normal(0, 0.1, V_init2.shape)
                W_init2 += np.random.normal(0, 0.1, W_init2.shape)
                
                params2 = np.concatenate([U_init2.flatten(), V_init2.flatten(), W_init2.flatten()])
                
                try:
                    res2 = minimize(error_func, params2, args=(rank,), 
                                 jac=lambda p, r: compute_gradient(p, r),
                                 method='L-BFGS-B', 
                                 options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
                    
                    if res2.fun < 1e-8:
                        U_final2 = res2.x[:rank * n * m].reshape((n * m, rank))
                        V_final2 = res2.x[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
                        W_final2 = res2.x[rank * n * m + rank * m * p :].reshape((p * n, rank))
                        
                        U_r2, V_r2, W_r2, ok2 = verify_and_round(U_final2, V_final2, W_final2)
                        if ok2:
                            best_result = {
                                "U": U_r2,
                                "V": V_r2,
                                "W": W_r2
                            }
                            return best_result
                except Exception:
                    pass

    return best_result
# EVOLVE-BLOCK-END
