import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n
    
    # Build the target tensor T
    T = np.zeros((u_rows, v_rows, w_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    T_flat = T.flatten()

    def compute_residual(U, V, W):
        # U: u_rows x R
        # V: v_rows x R
        # W: w_rows x R
        # Compute sum_c U[u,c] V[v,c] W[w,c]
        try:
            calc = np.einsum('uc,vc,wc->uvw', U, V, W)
        except Exception:
            calc = np.zeros_like(T)
            for c in range(U.shape[1]):
                calc += np.einsum('u,v,w->uvw', U[:,c], V[:,c], W[:,c])
        return (calc - T).flatten()

    def half_integer_penalty(U, V, W):
        # Penalty for not being half-integers
        # dist(x, 0.5 * Z) = min(0.5, |x - round(2x)/2|)
        # Actually, just distance to nearest half integer
        def dist_to_half(x):
            return np.abs(x - np.round(x * 2) / 2)
        
        p_u = np.sum(dist_to_half(U)**2)
        p_v = np.sum(dist_to_half(V)**2)
        p_w = np.sum(dist_to_half(W)**2)
        return p_u + p_v + p_w

    def cost_func(params, penalty_weight):
        R = len(params) // (u_rows + v_rows + w_rows)
        if R == 0:
            return 0.0
        u_flat, v_flat, w_flat = params[:u_rows*R], params[u_rows*R:u_rows*R + v_rows*R], params[u_rows*R + v_rows*R:]
        U = u_flat.reshape((u_rows, R))
        V = v_flat.reshape((v_rows, R))
        W = w_flat.reshape((w_rows, R))
        
        res = compute_residual(U, V, W)
        data_cost = np.sum(res**2)
        
        pen_cost = half_integer_penalty(U, V, W)
        
        return data_cost + penalty_weight * pen_cost

    def check_solution(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        try:
            calc = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        except Exception:
            return False, None, None, None

        if not np.allclose(calc, T, atol=1e-9):
            return False, None, None, None
            
        return True, U_r, V_r, W_r

    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety/cleanup
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Initialize with schoolbook solution as fallback
    U_school = np.zeros((u_rows, best_rank))
    V_school = np.zeros((v_rows, best_rank))
    W_school = np.zeros((w_rows, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower ranks
    # Start from 44 down to 30
    ranks_to_try = list(range(44, 29, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        max_restarts = 5
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(1234 + restart * 100 + R)
            
            # Random initialization
            # Try to initialize close to half-integers?
            # Or random normal?
            # Let's try random normal, but scaled.
            scale = 1.0 / np.sqrt(R)
            U_init = np.random.randn(u_rows, R) * scale
            V_init = np.random.randn(v_rows, R) * scale
            W_init = np.random.randn(w_rows, R) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B for better handling of the penalty
            # Penalty weight should be significant but not dominant initially
            # We can anneal it, but for simplicity, fixed weight
            
            # We want to find a solution where data_cost is 0.
            # The penalty helps guide to half-integers.
            
            try:
                # Use minimize with L-BFGS-B
                # We need to be careful with the number of function evaluations
                res = minimize(cost_func, params_init, args=(10.0,), method='L-BFGS-B', 
                               options={'maxiter': 1000, 'maxfun': 2000})
                
                # Extract solution
                u_flat, v_flat, w_flat = res.x[:u_rows*R], res.x[u_rows*R:u_rows*R + v_rows*R], res.x[u_rows*R + v_rows*R:]
                U_sol = u_flat.reshape((u_rows, R))
                V_sol = v_flat.reshape((v_rows, R))
                W_sol = w_flat.reshape((w_rows, R))
                
                # Check if it's a valid half-integer solution
                valid, U_r, V_r, W_r = check_solution(U_sol, V_sol, W_sol)
                
                if valid:
                    best_rank = R
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    # Found a solution for this rank, try lower rank
                    break 
            except Exception:
                pass
        
        # If we found a solution at R, we can stop or continue to lower?
        # The loop continues to lower R.
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
