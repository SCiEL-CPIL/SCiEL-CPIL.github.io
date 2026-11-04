import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    def compute_residual(U, V, W):
        res = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j_prime in range(m):
                        target = 1.0 if j == j_prime else 0.0
                        actual = 0.0
                        for c in range(U.shape[1]):
                            actual += U[i * m + j, c] * V[j_prime * p + k, c] * W[k * n + i, c]
                        res += (actual - target) ** 2
        return res

    def residual(params, r):
        if time.time() > deadline:
            return 1e10
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r + mp*r].reshape(mp, r)
        W = params[nm*r + mp*r:].reshape(pn, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.zeros((n, m, m, p))
        for c in range(r):
            u_c = U_3d[:, :, c]
            v_c = V_3d[:, :, c]
            w_c = W_3d[:, :, c]
            term = u_c[:, :, None, None] * v_c[None, None, :, :]
            term *= w_c.T[:, None, None, :]
            T += term
        
        J = np.eye(m)
        T_target = np.ones((n, 1, 1, p)) * J[None, :, :, None]
        
        res = np.sum((T - T_target) ** 2)
        return res

    # Try ranks from 21 upwards
    # Known lower bound is 21.
    ranks_to_try = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    max_restarts_per_rank = 10
    max_iter = 200
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        found = False
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline - 2:
                break
                
            np.random.seed(restart * 1000 + r)
            # Initialize with small integers to help rounding
            U_init = np.random.randint(-1, 2, size=(nm, r)).astype(float)
            V_init = np.random.randint(-1, 2, size=(mp, r)).astype(float)
            W_init = np.random.randint(-1, 2, size=(pn, r)).astype(float)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(residual, params_init, args=(r,), method='L-BFGS-B', 
                                  options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-8})
                
                if result.fun < 1e-6:
                    U_sol = result.x[:nm*r].reshape(nm, r)
                    V_sol = result.x[nm*r:nm*r + mp*r].reshape(mp, r)
                    W_sol = result.x[nm*r + mp*r:].reshape(pn, r)
                    
                    # Try rounding to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    res_round = compute_residual(U_round, V_round, W_round)
                    
                    if res_round < 1e-6:
                        best_rank = r
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
