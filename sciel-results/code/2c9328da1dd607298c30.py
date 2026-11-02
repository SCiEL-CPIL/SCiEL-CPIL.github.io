"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 24
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def check_half_integer(val, tol=1e-9):
        return abs(val * 2 - round(val * 2)) < tol
    
    def verify_solution(U, V, W, r, tol=1e-9):
        # Check half-integer constraint
        for c in range(r):
            for idx in range(nm):
                if not check_half_integer(U[idx][c]):
                    return False
            for idx in range(mp):
                if not check_half_integer(V[idx][c]):
                    return False
            for idx in range(pn):
                if not check_half_integer(W[idx][c]):
                    return False
        # Check exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def error_func(params, r):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp : r * nm + r * mp + r * pn]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    err += (s - 1.0) ** 2
        return err
    
    # Try ranks from 23 down to 18
    for r in range(rank_sb - 1, 17, -1):
        if time.time() - start_time > deadline:
            break
        
        total_vars = r * (nm + mp + pn)
        
        def objective_with_penalty(params, r):
            U_flat = params[:r * nm]
            V_flat = params[r * nm : r * nm + r * mp]
            W_flat = params[r * nm + r * mp : r * nm + r * mp + r * pn]
            
            U = U_flat.reshape(nm, r)
            V = V_flat.reshape(mp, r)
            W = W_flat.reshape(pn, r)
            
            # Tensor error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = 0.0
                        for c in range(r):
                            s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                        err += (s - 1.0) ** 2
            
            # Half-integer penalty
            pen = 0.0
            # Use vectorized operations for speed
            U_2 = U * 2
            V_2 = V * 2
            W_2 = W * 2
            
            U_diff = U_2 - np.round(U_2)
            V_diff = V_2 - np.round(V_2)
            W_diff = W_2 - np.round(W_2)
            
            pen = np.sum(U_diff**2) + np.sum(V_diff**2) + np.sum(W_diff**2)
            
            return err + 100.0 * pen
        
        # Multiple random attempts
        # Increase attempts since L-BFGS-B is faster
        for attempt in range(20):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(attempt * 1000 + r)
            # Initialize with small values to help convergence to sparse/structured solutions?
            x0 = np.random.uniform(-0.5, 0.5, size=total_vars)
            
            try:
                res = minimize(objective_with_penalty, x0, args=(r,), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            except Exception:
                continue
            
            # Check if the penalty is low enough (i.e., close to half-integers)
            if res.fun < 1e-6:
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp : r * nm + r * mp + r * pn]
                
                U = U_flat.reshape(nm, r)
                V = V_flat.reshape(mp, r)
                W = W_flat.reshape(pn, r)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                
                if verify_solution(U_list, V_list, W_list, r):
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found solution for this rank, move to lower
    
    return best_result
# EVOLVE-BLOCK-END
