import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    def check_decomposition(U, V, W, tol=1e-8):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def objective(params):
        # params is a flattened array of U, V, W
        r = int(len(params) / (nm + mp + pn))
        if r == 0: return 0
        U_flat, V_flat, W_flat = params[:r*nm], params[r*nm:r*nm+r*mp], params[r*nm+r*mp:]
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        W = W_flat.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.sum((T - 1.0)**2)

    def try_rank(target_rank, num_trials=3):
        if time.time() > deadline:
            return None
            
        nm_mp_pn = nm + mp + pn
        total_vars = target_rank * nm_mp_pn
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
                
            # Random initialization
            # Scale is important. 1/sqrt(rank) is a good guess.
            init_scale = 1.0 / np.sqrt(target_rank)
            x0 = np.random.randn(total_vars) * init_scale
            
            # Optimize
            try:
                res = opt.minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 200, 'ftol': 1e-10})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Extract U, V, W
                U_flat, V_flat, W_flat = res.x[:target_rank*nm], res.x[target_rank*nm:target_rank*nm+target_rank*mp], res.x[target_rank*nm+target_rank*mp:]
                U = U_flat.reshape(nm, target_rank)
                V = V_flat.reshape(mp, target_rank)
                W = W_flat.reshape(pn, target_rank)
                
                # Try to snap to half-integers
                U_snapped = 0.5 * np.round(2 * U)
                V_snapped = 0.5 * np.round(2 * V)
                W_snapped = 0.5 * np.round(2 * W)
                
                if check_decomposition(U_snapped, V_snapped, W_snapped):
                    return U_snapped, V_snapped, W_snapped
                
                # If snapping failed, maybe the solution is integer?
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                    
        return None

    # Search ranks from lower bound to schoolbook
    # Lower bound is max(nm, mp, pn) = 25
    # We can skip some if time is tight, but let's try a few
    # To be efficient, we might jump. But linear search is safer for "minimal".
    # Let's try 25 to 75.
    
    # Optimization: If we find a solution at R, we stop.
    # But low R is hard.
    # Let's try a few strategic ranks first? No, "minimal" is required.
    # We must find the smallest R.
    
    # To speed up, we can reduce trials for very low ranks if they fail quickly.
    for target_rank in range(25, rank_sb + 1):
        if time.time() > deadline:
            break
            
        # Use fewer trials for lower ranks to save time for higher ones if lower fail?
        # Or more trials for lower ranks because they are harder?
        # Let's use 3 trials for all.
        result = try_rank(target_rank, num_trials=3)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break # Found minimal rank, stop
            
    # If no better rank found, return schoolbook
    if best_rank == rank_sb:
         best_U, best_V, best_W = U_sb, V_sb, W_sb

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
