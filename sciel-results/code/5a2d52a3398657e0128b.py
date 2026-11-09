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
    # T[u, v, w] = 1 if u%3 == v//3 and v%3 == w//3 and w%3 == u%3
    target_flat = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_flat[u, v, w] = 1.0
    target_flat_list = target_flat.flatten()
    
    def compute_tensor(U, V, W):
        """Fast vectorized tensor computation: T = sum_r U_r * V_r * W_r"""
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify_and_round(U, V, W):
        """Try multiple rounding strategies and verify"""
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_flat_list
        err = np.dot(diff, diff)
        
        # If the continuous solution is not good, rounding won't help
        if err > 1e-4:
            return None, None, None, False
        
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Verify rounded
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
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_result = {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }

    # Try to find lower ranks using L-BFGS-B
    # Ranks to try: 24, 25, 26
    # Time allocation: prioritize lower ranks
    
    for rank in [24, 25, 26]:
        if time.time() > deadline - 20:
            break
            
        # Budget per rank
        budget = 10.0 if rank == 24 else 5.0
        rank_start = time.time()
        
        # Multiple random starts
        for trial in range(5):
            if time.time() - rank_start > budget:
                break
            if time.time() > deadline - 5:
                break
                
            np.random.seed(trial * 100 + rank * 7)
            
            # Initialize
            init_U = np.random.normal(0, 1.0, (n * m, rank))
            init_V = np.random.normal(0, 1.0, (m * p, rank))
            init_W = np.random.normal(0, 1.0, (p * n, rank))
            
            params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            def error_func(params):
                U = params[:rank * n * m].reshape((n * m, rank))
                V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
                W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
                T_calc = compute_tensor(U, V, W)
                diff = T_calc.flatten() - target_flat_list
                err = np.dot(diff, diff)
                # Add penalty for non-half-integer values to guide search
                pen = 0.1 * (np.sum((U * 2 - np.round(U * 2)) ** 2) +
                             np.sum((V * 2 - np.round(V * 2)) ** 2) +
                             np.sum((W * 2 - np.round(W * 2)) ** 2))
                return err + pen
            
            def grad_func(params):
                U = params[:rank * n * m].reshape((n * m, rank))
                V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
                W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
                
                T_calc = compute_tensor(U, V, W)
                diff = (T_calc.flatten() - target_flat_list).reshape((n * m, m * p, p * n))
                
                grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
                grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
                grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
                
                # Gradient of penalty
                pen_weight = 0.1
                grad_U += pen_weight * 4 * (U * 2 - np.round(U * 2))
                grad_V += pen_weight * 4 * (V * 2 - np.round(V * 2))
                grad_W += pen_weight * 4 * (W * 2 - np.round(W * 2))
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

            res = minimize(error_func, params, jac=grad_func, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 200})
            
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

    return best_result
# EVOLVE-BLOCK-END
