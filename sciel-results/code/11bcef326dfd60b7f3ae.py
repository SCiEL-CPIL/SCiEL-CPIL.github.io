import numpy as np
from scipy.optimize import minimize
import time
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 9
    dim_V = m * p  # 12
    dim_W = p * n  # 12
    
    # Build target tensor
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Schoolbook fallback
    rank_sb = n * m * p  # 36
    U_sb = np.zeros((dim_U, rank_sb))
    V_sb = np.zeros((dim_V, rank_sb))
    W_sb = np.zeros((dim_W, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx, col] = 1
                V_sb[v_idx, col] = 1
                W_sb[w_idx, col] = 1
                col += 1
    
    def verify_exact(U, V, W):
        """Exact integer verification: multiply by 2, round, check."""
        U2 = np.round(U * 2)
        V2 = np.round(V * 2)
        W2 = np.round(W * 2)
        
        if not (np.allclose(U2, np.round(U2), atol=1e-6) and
                np.allclose(V2, np.round(V2), atol=1e-6) and
                np.allclose(W2, np.round(W2), atol=1e-6)):
            return False
        
        U_int = np.round(U2).astype(np.int64)
        V_int = np.round(V2).astype(np.int64)
        W_int = np.round(W2).astype(np.int64)
        
        Rec = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
        T_scaled = (T * 8).astype(np.int64)
        
        return np.array_equal(Rec, T_scaled)
    
    def error_func(params, k):
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err
    
    start_time = time.time()
    max_time = 110  # Safety margin
    
    # Try ranks from low to high, with multiple restarts
    # Lower bound: max(dim_U, dim_V, dim_W) = 12
    # But realistically, we need at least enough to cover all 36 entries
    # Try ranks 20-35
    
    best_result = None
    best_rank = rank_sb
    
    # Ranks to try, with restart counts
    rank_schedule = [(20, 8), (22, 10), (24, 12), (25, 15), (26, 15), 
                     (27, 12), (28, 10), (29, 8), (30, 6), (31, 4), 
                     (32, 3), (33, 2), (34, 2), (35, 1)]
    
    for rank, restarts in rank_schedule:
        if time.time() - start_time > max_time:
            break
        if rank >= best_rank:
            continue
            
        for restart in range(restarts):
            if time.time() - start_time > max_time:
                break
                
            seed_val = 1000 * rank + restart * 137 + 42
            np.random.seed(seed_val)
            
            # Structured initialization: use small integers with some structure
            # Pattern: try to exploit the tensor structure
            init_U = np.zeros((dim_U, rank))
            init_V = np.zeros((dim_V, rank))
            init_W = np.zeros((dim_W, rank))
            
            # Fill with random small integers
            for c in range(rank):
                # Each column gets a random pattern
                u_vals = np.random.randint(-2, 3, size=dim_U)
                v_vals = np.random.randint(-2, 3, size=dim_V)
                w_vals = np.random.randint(-2, 3, size=dim_W)
                
                # Add some structure: make certain rows/columns dominant
                if c % 3 == 0:
                    u_vals[:3] = np.random.randint(1, 3, size=3)
                elif c % 3 == 1:
                    v_vals[:4] = np.random.randint(1, 3, size=4)
                else:
                    w_vals[:3] = np.random.randint(1, 3, size=3)
                
                init_U[:, c] = u_vals
                init_V[:, c] = v_vals
                init_W[:, c] = w_vals
            
            # Add small perturbation
            init_U += np.random.randn(dim_U, rank) * 0.05
            init_V += np.random.randn(dim_V, rank) * 0.05
            init_W += np.random.randn(dim_W, rank) * 0.05
            
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # Use Nelder-Mead for better local convergence
                res = minimize(error_func, params0, args=(rank,), method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                
                p_best = res.x
                U_opt = p_best[:dim_U * rank].reshape((dim_U, rank))
                V_opt = p_best[dim_U * rank : dim_U * rank + dim_V * rank].reshape((dim_V, rank))
                W_opt = p_best[dim_U * rank + dim_V * rank :].reshape((dim_W, rank))
                
                # Quick error check
                Rec_opt = np.einsum('ic,jc,kc->ijk', U_opt, V_opt, W_opt)
                err_opt = np.sum((Rec_opt - T) ** 2)
                
                if err_opt < 1e-6:
                    # Try rounding to half-integers
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify_exact(U_round, V_round, W_round):
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        
            except Exception:
                pass
    
    # Fallback: return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
