import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import lstsq

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # 150s limit, keep 10s margin
    deadline = start_time + 140.0

    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6

    # Precompute indices for the 18 equations
    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                eqs.append((ui, vi, wi))

    def calculate_error(params, R):
        # params: flat array
        # U: 6*R, V: 9*R, W: 6*R
        # We fix U[0,0] = 1.0 to break symmetry.
        # So we optimize U[1:]
        # Total vars: (6*R - 1) + 9*R + 6*R = 21*R - 1
        
        # Unpack
        u_flat = params[:6*R - 1]
        v_flat = params[6*R - 1:15*R - 1]
        w_flat = params[15*R - 1:]
        
        U_flat_full = np.empty(6*R)
        U_flat_full[0] = 1.0
        U_flat_full[1:] = u_flat
        U = U_flat_full.reshape(6, R)
        
        V = v_flat.reshape(9, R)
        W = w_flat.reshape(6, R)
        
        error = 0.0
        for ui, vi, wi in eqs:
            s = np.sum(U[ui] * V[vi] * W[wi])
            error += (s - 1.0)**2
        return error

    def snap_and_check(params, R):
        u_flat = params[:6*R - 1]
        v_flat = params[6*R - 1:15*R - 1]
        w_flat = params[15*R - 1:]
        
        U_flat_full = np.empty(6*R)
        U_flat_full[0] = 1.0
        U_flat_full[1:] = u_flat
        U = U_flat_full.reshape(6, R)
        
        V = v_flat.reshape(9, R)
        W = w_flat.reshape(6, R)
        
        # Snap to half-integers
        U_int = np.round(2 * U)
        V_int = np.round(2 * V)
        W_int = np.round(2 * W)
        
        # Check exactness: sum U_int * V_int * W_int == 8
        valid = True
        for ui, vi, wi in eqs:
            s = np.sum(U_int[ui] * V_int[vi] * W_int[wi])
            if s != 8:
                valid = False
                break
        
        if valid:
            return U_int / 2.0, V_int / 2.0, W_int / 2.0
        return None

    # Fallback schoolbook solution
    rank_school = n * m * p
    U_s = np.zeros((6, rank_school))
    V_s = np.zeros((9, rank_school))
    W_s = np.zeros((6, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    best_solution = (U_s, V_s, W_s)
    best_rank = rank_school

    # Search for lower rank
    # Prioritize R=11 (known lower bound for 2x3x3 is 11? Actually 11 is a known candidate)
    # Theoretical lower bound for 2x3x3 is 11 (Strassen-like).
    # We try R from 11 up to 17.
    for R in range(11, rank_school):
        if time.time() > deadline:
            break
            
        # Allocate time based on R
        # R=11 is most critical
        if R == 11:
            max_time_for_R = 60.0
            max_restarts = 50
        elif R == 12:
            max_time_for_R = 30.0
            max_restarts = 30
        else:
            max_time_for_R = 10.0
            max_restarts = 10
            
        R_start_time = time.time()
        found = False
        
        for restart in range(max_restarts):
            if time.time() > R_start_time + max_time_for_R:
                break
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Small values to help convergence
            u_init = np.random.uniform(-0.5, 0.5, 6*R - 1)
            v_init = np.random.uniform(-0.5, 0.5, 9*R)
            w_init = np.random.uniform(-0.5, 0.5, 6*R)
            
            params = np.concatenate([u_init, v_init, w_init])
            
            # Use Nelder-Mead for speed
            res = minimize(calculate_error, params, args=(R,), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e-5:
                # Polish with Powell
                res2 = minimize(calculate_error, res.x, args=(R,), method='Powell', 
                                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res2.fun < res.fun:
                    res = res2
                    
                if res.fun < 1e-6:
                    sol = snap_and_check(res.x, R)
                    if sol is not None:
                        best_rank = R
                        best_solution = sol
                        found = True
                        break
        
        if found:
            # If we found R=11, we can stop or continue to see if lower (unlikely)
            if R == 11:
                break

    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
