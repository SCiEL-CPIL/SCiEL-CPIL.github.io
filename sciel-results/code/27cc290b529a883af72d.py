import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find lower rank
    # Ranks to try: 50, 40, 30
    for r in [50, 40, 30]:
        if time.time() > deadline:
            break
            
        # Random initialization
        np.random.seed(42 + r)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Optimization
        # Use a simple gradient descent or scipy
        # Given time, try scipy.minimize
        from scipy.optimize import minimize
        
        def objective(params):
            U_flat = params[:n*m*r]
            V_flat = params[n*m*r : n*m*r + m*p*r]
            W_flat = params[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r]
            
            U_curr = U_flat.reshape(n*m, r)
            V_curr = V_flat.reshape(m*p, r)
            W_curr = W_flat.reshape(p*n, r)
            
            # Compute tensor
            # T_approx = sum_c U_c (x) V_c (x) W_c
            # Efficiently: np.einsum
            T_approx = np.einsum('ac,bc,dc->abd', U_curr, V_curr, W_curr)
            diff = T - T_approx
            return np.sum(diff**2)
            
        def gradient(params):
            U_flat = params[:n*m*r]
            V_flat = params[n*m*r : n*m*r + m*p*r]
            W_flat = params[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r]
            
            U_curr = U_flat.reshape(n*m, r)
            V_curr = V_flat.reshape(m*p, r)
            W_curr = W_flat.reshape(p*n, r)
            
            T_approx = np.einsum('ac,bc,dc->abd', U_curr, V_curr, W_curr)
            diff = T - T_approx
            
            # Gradient w.r.t U
            # dL/dU_ac = -2 * sum_bd diff_abd V_bc W_dc
            grad_U = -2 * np.einsum('abd,bc,dc->ac', diff, V_curr, W_curr)
            grad_V = -2 * np.einsum('abd,ac,dc->bc', diff, U_curr, W_curr)
            grad_W = -2 * np.einsum('abd,ac,bc->dc', diff, U_curr, V_curr)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
        params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Run optimization
        # Use L-BFGS-B
        try:
            res = minimize(objective, params_init, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
            
        if res.success or res.fun < 1e-6:
            params_opt = res.x
            U_opt = params_opt[:n*m*r].reshape(n*m, r)
            V_opt = params_opt[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_opt = params_opt[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r].reshape(p*n, r)
            
            # Round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            # Verify
            T_check = np.einsum('ac,bc,dc->abd', U_round, V_round, W_round)
            if np.allclose(T_check, T, atol=1e-9):
                best_rank = r
                best_U = U_round
                best_V = V_round
                best_W = W_round
                break # Found a good one, stop searching lower? Or continue?
                # Continue to find even lower?
                # If 50 works, try 40.
        
    # Convert to lists of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
