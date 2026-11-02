import time
import numpy as np

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

def _compute_residual(U, V, W, T_true):
    """Compute the maximum absolute residual."""
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(M - T_true))

def _als_rank31(U0, V0, W0, T_true, n, m, p, max_iters=200, tol=1e-12, seed=0):
    """Alternating least squares for rank-31 tensor decomposition."""
    rng = np.random.default_rng(seed)
    nm, mp, pn = n * m, m * p, p * n
    rank = 31
    
    # Initialize by dropping a random column from rank-32 and adding small noise
    drop_col = rng.integers(0, 32)
    mask = [i for i in range(32) if i != drop_col]
    U = U0[:, mask].copy()
    V = V0[:, mask].copy()
    W = W0[:, mask].copy()
    
    # Add small random perturbation
    U += rng.normal(0, 0.01, size=U.shape)
    V += rng.normal(0, 0.01, size=V.shape)
    W += rng.normal(0, 0.01, size=W.shape)
    
    prev_loss = float('inf')
    
    for it in range(max_iters):
        # Update U: for each i, solve U[i,:] * (V^T * W_reshaped) = T_true[i,:,:]
        for i in range(nm):
            # Compute the right-hand side and Gram matrix
            # T_true[i,:,:] has shape (mp, pn)
            # We need: U[i,:] @ V^T @ W_reshaped = T_true[i,:,:]
            # Let A = V^T @ W_reshaped (shape rank x pn), B = T_true[i,:,:] (shape mp x pn)
            # Then U[i,:] @ V^T = B @ W^{-1}... 
            # Actually: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # For fixed i: sum_r U[i,r] * (V[j,r] * W[k,r]) = T[i,j,k]
            # Let C[j,k] = sum_r V[j,r] * W[k,r]  (this is V @ W^T)
            # Then: sum_r U[i,r] * V[j,r] * W[k,r] is NOT simply U[i,:] @ C
            # 
            # Correct approach: 
            # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # This is a 3-way product. For ALS, we fix two factors and solve for the third.
            # Fix V and W, solve for U:
            # For each i: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k] for all j,k
            # Let's define: for each r, the outer product v_r * w_r^T has shape (mp, pn)
            # Then T[i,:,:] = sum_r U[i,r] * (v_r * w_r^T)
            # This is a linear system in U[i,:]
            
            # Build the design matrix: each column r corresponds to vec(v_r * w_r^T)
            # But that's mp*pn dimensional, too large. Instead, use the fact that
            # we only need to satisfy T[i,j,k] for the specific (j,k) pairs where T=1.
            
            pass  # This approach is getting complex; let me use a simpler formulation
        
        # Simpler: use the full tensor equation
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For fixed V,W: this is linear in U[i,r]
        # The constraint is: for each (i,j,k) where T[i,j,k]=1:
        #   sum_r U[i,r] * V[j,r] * W[k,r] = 1
        # For (i,j,k) where T[i,j,k]=0:
        #   sum_r U[i,r] * V[j,r] * W[k,r] = 0
        
        # This is a linear system. Let's build it properly.
        # Actually, let's just use scipy minimize with a better objective.
        break
    
    # Fallback: use scipy minimize
    from scipy.optimize import minimize
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U_m = U_flat.reshape(nm, rank)
        V_m = V_flat.reshape(mp, rank)
        W_m = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
        residual = M - T_true
        return np.sum(residual**2)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        
        if result.fun < 1e-10:
            U_flat = result.x[:nm * rank]
            V_flat = result.x[nm * rank : nm * rank + mp * rank]
            W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U_sol = U_flat.reshape(nm, rank)
            V_sol = V_flat.reshape(mp, rank)
            W_sol = W_flat.reshape(pn, rank)
            return U_sol, V_sol, W_sol, result.fun
    except Exception:
        pass
    
    return None

def _try_rank31_search(U32, V32, W32, T_true, n, m, p, time_budget):
    """Try to find a rank-31 decomposition using multiple restarts."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank31 = 31
    
    best_result = None
    best_loss = float('inf')
    
    rng = np.random.default_rng(RNG_SEED)
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget * 0.85:
            break
        
        # Create different initializations
        drop_col = restart % 32
        mask = [i for i in range(32) if i != drop_col]
        U_init = U32[:, mask].copy() + rng.normal(0, 0.05, size=(nm, rank31))
        V_init = V32[:, mask].copy() + rng.normal(0, 0.05, size=(mp, rank31))
        W_init = W32[:, mask].copy() + rng.normal(0, 0.05, size=(pn, rank31))
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        from scipy.optimize import minimize
        
        def objective(params):
            U_flat = params[:nm * rank31]
            V_flat = params[nm * rank31 : nm * rank31 + mp * rank31]
            W_flat = params[nm * rank31 + mp * rank31 : nm * rank31 + mp * rank31 + pn * rank31]
            
            U_m = U_flat.reshape(nm, rank31)
            V_m = V_flat.reshape(mp, rank31)
            W_m = W_flat.reshape(pn, rank31)
            
            M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
            residual = M - T_true
            return np.sum(residual**2)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
            
            if result.fun < 1e-10 and result.fun < best_loss:
                U_flat = result.x[:nm * rank31]
                V_flat = result.x[nm * rank31 : nm * rank31 + mp * rank31]
                W_flat = result.x[nm * rank31 + mp * rank31 : nm * rank31 + mp * rank31 + pn * rank31]
                U_sol = U_flat.reshape(nm, rank31)
                V_sol = V_flat.reshape(mp, rank31)
                W_sol = W_flat.reshape(pn, rank31)
                
                # Verify with max norm
                M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
                max_res = np.max(np.abs(M - T_true))
                
                if max_res < 1e-8:
                    best_loss = result.fun
                    best_result = (U_sol.copy(), V_sol.copy(), W_sol.copy(), max_res)
        except Exception:
            continue
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Precompute true tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    # Try to find rank-31 using optimization with multiple restarts
    time_remaining = deadline - time.time()
    if time_remaining > 20.0:
        result = _try_rank31_search(U32, V32, W32, T_true, n, m, p, time_remaining - 5.0)
        if result is not None:
            U_sol, V_sol, W_sol, max_res = result
            # Final verification
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            residual = np.max(np.abs(M - T_true))
            if residual < 1e-6:
                return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
