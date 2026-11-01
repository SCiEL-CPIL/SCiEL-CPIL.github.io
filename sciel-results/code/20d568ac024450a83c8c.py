import time
import numpy as np
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 2, 9]
RNG_SEED = 42

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _build_rank31_init_merge(U32, V32, W32, i, j):
    """Create a rank-31 initialization by merging columns i and j."""
    rank = 32
    indices = [k for k in range(rank) if k != j]
    # We keep column i and remove j.
    # We modify column i to be the average of i and j.
    U31 = np.zeros((U32.shape[0], 31))
    V31 = np.zeros((V32.shape[0], 31))
    W31 = np.zeros((W32.shape[0], 31))
    
    col_idx = 0
    for k in indices:
        if k == i:
            U31[:, col_idx] = (U32[:, i] + U32[:, j]) / 2.0
            V31[:, col_idx] = (V32[:, i] + V32[:, j]) / 2.0
            W31[:, col_idx] = (W32[:, i] + W32[:, j]) / 2.0
        else:
            U31[:, col_idx] = U32[:, k]
            V31[:, col_idx] = V32[:, k]
            W31[:, col_idx] = W32[:, k]
        col_idx += 1
        
    return U31, V31, W31

def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition using least_squares with merge initializations."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank31 = 31
    
    # Precompute true tensor flattened
    T_true = np.zeros((nm * mp * pn,))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                # Index in flattened tensor
                # T is (nm, mp, pn). Flatten C-order: index = (i * mp + j) * pn + k
                idx = (i * mp + j) * pn + k
                T_true[idx] = 1.0
    
    best_score = float('inf')
    best_result = None
    
    rng = np.random.default_rng(RNG_SEED)
    
    # Try merging pairs of columns
    # There are 32 columns. 32*31/2 = 496 pairs.
    # We can't try all. Let's pick a random subset or all if fast enough.
    # Each optimization might take ~1s. 150s budget. 
    # Let's try 50 random pairs and a few specific ones.
    
    pairs = []
    for i in range(32):
        for j in range(i+1, 32):
            pairs.append((i, j))
    
    rng.shuffle(pairs)
    # Limit number of tries to fit in time
    max_tries = 40
    
    for idx, (i, j) in enumerate(pairs[:max_tries]):
        if time.time() - start_time > time_budget * 0.9:
            break
            
        U31, V31, W31 = _build_rank31_init_merge(U0, V0, W0, i, j)
        
        x0 = np.concatenate([U31.flatten(), V31.flatten(), W31.flatten()])
        
        def residual_func(params):
            U_flat = params[:nm * rank31]
            V_flat = params[nm * rank31 : nm * rank31 + mp * rank31]
            W_flat = params[nm * rank31 + mp * rank31 : nm * rank31 + mp * rank31 + pn * rank31]
            
            U = U_flat.reshape(nm, rank31)
            V = V_flat.reshape(mp, rank31)
            W = W_flat.reshape(pn, rank31)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            M_flat = M.flatten()
            
            return M_flat - T_true
        
        try:
            result = least_squares(residual_func, x0, method='lm', 
                                   max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            
            if result.success or result.cost < 1e-8:
                x_sol = result.x
                U_flat = x_sol[:nm * rank31]
                V_flat = x_sol[nm * rank31 : nm * rank31 + mp * rank31]
                W_flat = x_sol[nm * rank31 + mp * rank31 : nm * rank31 + mp * rank31 + pn * rank31]
                U_sol = U_flat.reshape(nm, rank31)
                V_sol = V_flat.reshape(mp, rank31)
                W_sol = W_flat.reshape(pn, rank31)
                
                # Check max residual
                M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
                residual = np.max(np.abs(M - T_true.reshape(nm, mp, pn)))
                
                if residual < 1e-6 and residual < best_score:
                    best_score = residual
                    best_result = (U_sol.copy(), V_sol.copy(), W_sol.copy(), residual)
                    
                    # If we found a very good one, we can break or continue to polish
                    if residual < 1e-10:
                        break
                        
        except Exception:
            continue
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using direct optimization with multiple restarts
    time_remaining = deadline - time.time()
    if time_remaining > 30.0:
        result = _try_rank31_search(U32, V32, W32, n, m, p, time_remaining - 5.0)
        if result is not None:
            U_sol, V_sol, W_sol, score = result
            # Verify again just in case
            T_true = np.zeros((n * m, m * p, p * n))
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        i = a * m + b
                        j = b * p + c
                        k = c * n + a
                        T_true[i, j, k] = 1.0
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            residual = np.max(np.abs(M - T_true))
            if residual < 1e-6:
                return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
