"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    # Check off-diagonal entries are zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > 1e-9:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Schoolbook fallback
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Build target tensor once
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def _optimize_rank(target_rank, max_iters=100, seed=42):
        """Try to find a rank-target_rank decomposition using L-BFGS-B."""
        rng = np.random.RandomState(seed)
        
        # Initialize with structure-aware random init
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_x = x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W_x = x[2*n*m*target_rank:].reshape(p*n, target_rank)
            
            T = np.einsum('ar,br,cr->abc', U_x, V_x, W_x)
            diff = T - T_target
            return np.sum(diff * diff)
        
        def gradient(x):
            U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_x = x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W_x = x[2*n*m*target_rank:].reshape(p*n, target_rank)
            
            T = np.einsum('ar,br,cr->abc', U_x, V_x, W_x)
            diff = T - T_target
            
            grad_U = np.einsum('abc,br,cr->ar', diff, V_x, W_x)
            grad_V = np.einsum('abc,ar,cr->br', diff, U_x, W_x)
            grad_W = np.einsum('abc,ar,br->cr', diff, U_x, V_x)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        try:
            res = so.minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-8:
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[2*n*m*target_rank:].reshape(p*n, target_rank)
                
                # Snap to half-integers if close
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                err_snap = np.sum((T_snap - T_target)**2)
                
                if err_snap < 1e-9:
                    return U_snap, V_snap, W_snap, err_snap
                
                # If snap fails, return the optimized floats
                err = res.fun
                if err < 1e-10:
                     return U_opt, V_opt, W_opt, err
                     
        except Exception:
            pass
            
        return None, None, None, float('inf')
    
    # Try various ranks in increasing order
    # Lower bound for 2x3x7 is likely > 21. 
    # We will try ranks starting from 21 up to 30 first, then go higher if needed.
    ranks_to_try = list(range(21, 31)) + list(range(31, rank_school + 1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try a couple of seeds
        found = False
        for seed in [42, 123, 999]:
            if time.time() > deadline - 5:
                break
                
            U_opt, V_opt, W_opt, err = _optimize_rank(target_rank, max_iters=100, seed=seed)
            
            if U_opt is not None and err < 1e-9:
                if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                    best_rank = target_rank
                    best_result = {
                        "U": U_opt.tolist(),
                        "V": V_opt.tolist(),
                        "W": W_opt.tolist()
                    }
                    found = True
                    break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
