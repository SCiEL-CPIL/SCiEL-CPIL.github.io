import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Construct the target tensor
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Schoolbook rank
    rank_school = n * m * p
    
    # Time guard
    start_time = time.time()
    max_time = 140  # Leave 10s for safety
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Initial solution: Schoolbook
    U_init = np.zeros((dim_U, rank_school))
    V_init = np.zeros((dim_V, rank_school))
    W_init = np.zeros((dim_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_init[u_idx, col] = 1
                V_init[v_idx, col] = 1
                W_init[w_idx, col] = 1
                col += 1
    best_U, best_V, best_W = U_init, V_init, W_init

    def error_func(params, k):
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err

    def try_rank(k, max_restarts=5, max_iter_per_restart=300):
        """Try to find a rank-k decomposition with multiple restarts."""
        if time.time() - start_time > max_time:
            return None
        
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Different initialization strategies
            if restart == 0:
                # Sparse initialization with small values
                np.random.seed(1000 + k)
                init_U = np.random.randn(dim_U, k) * 0.3
                init_V = np.random.randn(dim_V, k) * 0.3
                init_W = np.random.randn(dim_W, k) * 0.3
            elif restart == 1:
                # Slightly larger scale
                np.random.seed(2000 + k)
                init_U = np.random.randn(dim_U, k) * 0.5
                init_V = np.random.randn(dim_V, k) * 0.5
                init_W = np.random.randn(dim_W, k) * 0.5
            elif restart == 2:
                # Even larger scale
                np.random.seed(3000 + k)
                init_U = np.random.randn(dim_U, k) * 0.8
                init_V = np.random.randn(dim_V, k) * 0.8
                init_W = np.random.randn(dim_W, k) * 0.8
            elif restart == 3:
                # Sparse: mostly zeros with a few non-zero entries
                np.random.seed(4000 + k)
                init_U = np.zeros((dim_U, k))
                init_V = np.zeros((dim_V, k))
                init_W = np.zeros((dim_W, k))
                for c in range(k):
                    # Randomly pick a few positions to initialize
                    u_pos = np.random.randint(0, dim_U, size=min(3, dim_U))
                    v_pos = np.random.randint(0, dim_V, size=min(3, dim_V))
                    w_pos = np.random.randint(0, dim_W, size=min(3, dim_W))
                    init_U[u_pos, c] = np.random.choice([-1, 1]) * 0.5
                    init_V[v_pos, c] = np.random.choice([-1, 1]) * 0.5
                    init_W[w_pos, c] = np.random.choice([-1, 1]) * 0.5
            else:
                # Mixed scale
                np.random.seed(5000 + k + restart)
                scale = 0.4 + 0.1 * restart
                init_U = np.random.randn(dim_U, k) * scale
                init_V = np.random.randn(dim_V, k) * scale
                init_W = np.random.randn(dim_W, k) * scale
            
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # Use L-BFGS-B with more iterations
                res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': max_iter_per_restart, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                
                # Extract best params
                p_best = res.x
                U_opt = p_best[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                # Round to nearest integer or half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    return (U_round, V_round, W_round)
                    
                # If close but not exact, try a high-precision polish
                err = np.sum((Rec_round - T) ** 2)
                if err < 1e-6:
                    # Try to refine with tighter tolerance
                    params_refined = np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()])
                    res2 = minimize(error_func, params_refined, args=(k,), method='L-BFGS-B', 
                                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                    p_ref = res2.x
                    U_ref = p_ref[:dim_U * k].reshape((dim_U, k))
                    V_ref = p_ref[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                    W_ref = p_ref[dim_U * k + dim_V * k :].reshape((dim_W, k))
                    
                    U_ref_round = np.round(U_ref * 2) / 2
                    V_ref_round = np.round(V_ref * 2) / 2
                    W_ref_round = np.round(W_ref * 2) / 2
                    
                    Rec_ref = np.einsum('ic,jc,kc->ijk', U_ref_round, V_ref_round, W_ref_round)
                    if np.allclose(Rec_ref, T, atol=1e-9):
                        return (U_ref_round, V_ref_round, W_ref_round)
                        
            except Exception as e:
                pass

        return None

    # Try ranks from a lower bound upwards
    # Known lower bounds suggest trying from around 22-25
    for k in range(22, rank_school):
        if time.time() - start_time > max_time:
            break
            
        result = try_rank(k, max_restarts=6, max_iter_per_restart=400)
        if result is not None:
            U_found, V_found, W_found = result
            return {"U": U_found.tolist(), "V": V_found.tolist(), "W": W_found.tolist()}

    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
