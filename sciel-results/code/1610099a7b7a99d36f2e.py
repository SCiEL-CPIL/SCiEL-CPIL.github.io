import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    trivial_rank = n * m * p  # 24
    
    # Trivial solution for 2x3x4
    U_trivial = np.zeros((nm, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((mp, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((pn, trivial_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    best_rank = trivial_rank
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    def check_valid(U, V, W, rank, tol=1e-9):
        # Check if U, V, W with given rank exactly decompose the 2x3x4 matrix multiplication tensor
        # n=2, m=3, p=4
        # U shape: (nm, rank) = (6, rank)
        # V shape: (mp, rank) = (12, rank)
        # W shape: (pn, rank) = (8, rank)
        # Tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should equal 1 for all i in [0,2), j in [0,3), k in [0,4)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    # Try to find a solution for 2x3x4 with ranks below 24
    # We'll try ranks from 18 up to 23
    for target_rank in range(18, 24):
        if time.time() > deadline - 30.0:
            break
        
        np.random.seed(42)
        U_init = np.random.rand(nm, target_rank) * 0.5 - 0.25
        V_init = np.random.rand(mp, target_rank) * 0.5 - 0.25
        W_init = np.random.rand(pn, target_rank) * 0.5 - 0.25
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def error_func(x):
            U = x[:nm*target_rank].reshape(nm, target_rank)
            V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                        err += (s - 1.0)**2
            return err
        
        try:
            res = minimize(error_func, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
            if res.success and res.fun < 1e-9:
                U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                U_s, V_s, W_s = snap_to_half_integer(U_sol, V_sol, W_sol)
                if check_valid(U_s, V_s, W_s, target_rank, tol=1e-9):
                    best_rank = target_rank
                    best_U = U_s
                    best_V = V_s
                    best_W = W_s
                    break
        except Exception:
            pass
            
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
