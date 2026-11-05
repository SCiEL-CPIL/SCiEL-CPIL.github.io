import numpy as np
import random
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    # T is a 3D array of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Flatten T for objective function
    T_flat = T_target.flatten()
    
    # Helper to check if a set of factors is a valid exact decomposition
    def is_valid_exact(U, V, W):
        # U, V, W are numpy arrays of shape (nm, R), (mp, R), (pn, R)
        # Check integer/half-integer
        U2 = 2 * U
        V2 = 2 * V
        W2 = 2 * W
        
        # Check if close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-6):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-6):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-6):
            return False
            
        # Check exact decomposition
        # Compute S[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can use einsum
        S = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Compare with T_target
        if not np.allclose(S, T_target, atol=1e-6):
            return False
            
        return True

    # Helper to construct schoolbook decomposition
    def schoolbook():
        R = nm * mp * pn
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for lower rank
    # We try ranks from a lower bound up to schoolbook
    # Known lower bound for 3x4x4 is likely > 10. Let's start from 15.
    # Time limit 120s.
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    # Optimization: Use scipy.optimize.least_squares
    try:
        from scipy.optimize import least_squares
    except ImportError:
        least_squares = None
        # Fallback to Nelder-Mead if least_squares not available
        try:
            from scipy.optimize import minimize
        except ImportError:
            minimize = None
    else:
        minimize = None # Prefer least_squares if available

    # If no scipy, we are in trouble for optimization. 
    # But standard library + numpy is available. 
    # The prompt says "NumPy and SciPy are available".

    best_result = None
    
    # Try ranks from 15 to 48
    # We can also try to go down from 48 if we find a solution, but finding the MINIMAL is the goal.
    # So we iterate R from low to high.
    
    for R in range(15, nm * mp * pn + 1):
        if time.time() - start_time > time_limit:
            break
            
        total_vars = R * (nm + mp + pn)
        
        # Try a few random initializations
        attempts = 10
        found_in_rank = False
        
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Random init
            params = np.random.normal(0.0, 1.0, total_vars)
            
            # Define residual function for least_squares
            def residual(params, R=R):
                size_u = nm * R
                size_v = mp * R
                size_w = pn * R
                
                U_flat = params[:size_u]
                V_flat = params[size_u:size_u+size_v]
                W_flat = params[size_u+size_v:]
                
                U = U_flat.reshape(nm, R)
                V = V_flat.reshape(mp, R)
                W = W_flat.reshape(pn, R)
                
                S = np.einsum('ar,br,cr->abc', U, V, W)
                return (S - T_target).flatten()

            if least_squares:
                res = least_squares(residual, params, method='lm', max_nfev=2000)
                # Check if residual is small
                if res.cost < 1e-8:
                    p_opt = res.x
                    U_opt = p_opt[:nm*R].reshape(nm, R)
                    V_opt = p_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = p_opt[nm*R+mp*R:].reshape(pn, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if is_valid_exact(U_round, V_round, W_round):
                        best_result = (R, U_round, V_round, W_round)
                        found_in_rank = True
                        break
            elif minimize:
                # Fallback
                def objective(params, R=R):
                    size_u = nm * R
                    size_v = mp * R
                    size_w = pn * R
                    U_flat = params[:size_u]
                    V_flat = params[size_u:size_u+size_v]
                    W_flat = params[size_u+size_v:]
                    U = U_flat.reshape(nm, R)
                    V = V_flat.reshape(mp, R)
                    W = W_flat.reshape(pn, R)
                    S = np.einsum('ar,br,cr->abc', U, V, W)
                    return np.sum((S - T_target)**2)
                
                res = minimize(objective, params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success or res.fun < 1e-8:
                    p_opt = res.x
                    U_opt = p_opt[:nm*R].reshape(nm, R)
                    V_opt = p_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = p_opt[nm*R+mp*R:].reshape(pn, R)
                    
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if is_valid_exact(U_round, V_round, W_round):
                        best_result = (R, U_round, V_round, W_round)
                        found_in_rank = True
                        break
                        
        if found_in_rank:
            break

    if best_result:
        R, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
