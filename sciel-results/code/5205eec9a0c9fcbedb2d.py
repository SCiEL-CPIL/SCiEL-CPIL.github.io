import numpy as np
from scipy.optimize import minimize
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    rank_school = n * m * p
    
    # Schoolbook solution
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
                
    best_rank = rank_school
    best_U, best_V, best_W = U_init, V_init, W_init

    start_time = time.time()
    max_time = 140  # Safety margin

    def verify_decomposition(U, V, W):
        """Verify exact decomposition using integer arithmetic."""
        # Convert to integers by multiplying by 2
        U2 = np.round(U * 2)
        V2 = np.round(V * 2)
        W2 = np.round(W * 2)
        
        # Check if they're close to integers
        if not (np.allclose(U2, np.round(U2), atol=1e-8) and
                np.allclose(V2, np.round(V2), atol=1e-8) and
                np.allclose(W2, np.round(W2), atol=1e-8)):
            return False
        
        U_int = np.round(U2).astype(np.int64)
        V_int = np.round(V2).astype(np.int64)
        W_int = np.round(W2).astype(np.int64)
        
        # Compute reconstruction: sum_c U[:,c] * V[:,c] * W[:,c]
        # Using einsum with integer arithmetic
        Rec = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
        
        # Target in doubled space: T * 8 (since each product of 3 half-integers gives /8)
        # Actually: U = U_int/2, V = V_int/2, W = W_int/2
        # Rec = sum (U_int[c,i]/2) * (V_int[c,j]/2) * (W_int[c,k]/2) = sum(U_int*V_int*W_int)/8
        # So Rec*8 should equal T*8 = 8*T
        Rec_scaled = Rec  # This is sum(U_int*V_int*W_int)
        T_scaled = (T * 8).astype(np.int64)
        
        return np.array_equal(Rec_scaled, T_scaled)

    def error_func(params, k):
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err

    # Try to find low-rank decompositions
    # Known: 3x3x4 has rank at most 25 (from literature)
    # Try ranks in order of preference
    
    ranks_to_try = [25, 26, 27, 28, 29, 30]
    
    for k in ranks_to_try:
        if time.time() - start_time > max_time:
            break
            
        # Adaptive restarts: fewer for higher ranks since they're less likely to be optimal
        if k <= 25:
            restarts = 50
        elif k <= 27:
            restarts = 30
        else:
            restarts = 15
        
        for restart in range(restarts):
            if time.time() - start_time > max_time:
                break
                
            seed_val = 42 + k * 1000 + restart * 7
            np.random.seed(seed_val)
            
            # Smart initialization: start from structured patterns
            # Use small integer values in {-2, -1, 0, 1, 2}
            val_range = 2
            init_U = np.random.randint(-val_range, val_range + 1, size=(dim_U, k)).astype(float)
            init_V = np.random.randint(-val_range, val_range + 1, size=(dim_V, k)).astype(float)
            init_W = np.random.randint(-val_range, val_range + 1, size=(dim_W, k)).astype(float)
            
            # Add small perturbation to break symmetry
            init_U += np.random.randn(dim_U, k) * 0.01
            init_V += np.random.randn(dim_V, k) * 0.01
            init_W += np.random.randn(dim_W, k) * 0.01
            
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # Use Nelder-Mead for better local convergence on small problems
                # Or L-BFGS-B with fewer iterations
                res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6,
                                        'maxls': 20})
                
                p_best = res.x
                U_opt = p_best[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                # Quick check: is the error small enough to potentially round?
                Rec_opt = np.einsum('ic,jc,kc->ijk', U_opt, V_opt, W_opt)
                err_opt = np.sum((Rec_opt - T) ** 2)
                
                if err_opt < 1e-4:
                    # Try rounding to half-integers
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify_decomposition(U_round, V_round, W_round):
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        
            except Exception:
                pass

    # Fallback: return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
