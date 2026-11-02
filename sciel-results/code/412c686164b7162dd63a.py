"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target multiplication tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.float64)
    V_school = np.zeros((mp, rank_school), dtype=np.float64)
    W_school = np.zeros((pn, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_solution = {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }

    def tensor_from_factors(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def objective(params, R):
        total_dim = R * (nm + mp + pn)
        if len(params) != total_dim:
            return 1e10
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        T_calc = tensor_from_factors(U, V, W)
        diff = T_calc - T
        return np.sum(diff**2)

    def try_rank(R, max_iter=200):
        if time.time() > deadline - 15:
            return None
        
        num_vars = R * (nm + mp + pn)
        
        # Use a few restarts
        restarts = 5
        for r in range(restarts):
            if time.time() > deadline - 15:
                break
            
            x0 = np.random.randn(num_vars) * 0.1
            try:
                # Use L-BFGS-B
                res = minimize(objective, x0, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': max_iter, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < 1e-8:
                    U = res.x[:R*nm].reshape(nm, R)
                    V = res.x[R*nm:R*nm+R*mp].reshape(mp, R)
                    W = res.x[R*nm+R*mp:].reshape(pn, R)
                    
                    # Round to nearest half-integer
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify
                    T_check = tensor_from_factors(U_int, V_int, W_int)
                    if np.allclose(T_check, T, atol=1e-10):
                        return {
                            "U": U_int,
                            "V": V_int,
                            "W": W_int
                        }
            except Exception:
                pass
        return None

    # Search for lower rank
    # Start from 44 down to 30
    for R in range(rank_school - 1, 29, -1):
        if time.time() > deadline - 5:
            break
        
        sol = try_rank(R)
        if sol is not None:
            best_solution = sol
            # Continue searching for even lower rank
            pass

    return best_solution
# EVOLVE-BLOCK-END
