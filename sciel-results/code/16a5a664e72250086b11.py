import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the target tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = i * m + j
    # v = j * p + k
    # w = k * n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    T_norm = np.linalg.norm(T_flat)
    
    def compute_residuals(params):
        r = params.size // (dim_u + dim_v + dim_w)
        if r == 0:
            return np.zeros(T_flat.size)
        
        U = params[0:r*dim_u].reshape(r, dim_u).T # (dim_u, r)
        V = params[r*dim_u:r*(dim_u+dim_v)].reshape(r, dim_v).T # (dim_v, r)
        W = params[r*(dim_u+dim_v):].reshape(r, dim_w).T # (dim_w, r)
        
        P = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        return (P - T).flatten()

    def verify_solution(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        P = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        
        if np.allclose(P, T, atol=1e-9):
            return True, U_r, V_r, W_r
        return False, None, None

    # Schoolbook solution (Rank 30)
    schoolbook_rank = n * m * p
    U_school = np.zeros((dim_u, schoolbook_rank))
    V_school = np.zeros((dim_v, schoolbook_rank))
    W_school = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_solution = None
    best_rank = schoolbook_rank
    
    # Try ranks from 29 downwards
    # If 29 is possible, it's a 1-rank improvement over schoolbook
    ranks_to_try = [29, 28, 27, 26, 25]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        if r >= best_rank:
            continue
            
        # Number of restarts
        # We have ~100 seconds left.
        # Let's try 200 restarts for rank 29, 100 for 28, etc.
        max_restarts = 200 if r == 29 else 100
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            # Scale to keep values reasonable
            scale = 1.0 / np.sqrt(r)
            U_init = np.random.randn(dim_u, r) * scale
            V_init = np.random.randn(dim_v, r) * scale
            W_init = np.random.randn(dim_w, r) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use 'lm' for speed, it's good for unconstrained problems
                # max_nfev limited to prevent hanging
                res = least_squares(compute_residuals, params_init, method='lm', max_nfev=200)
                
                params_sol = res.x
                U_sol = params_sol[0:r*dim_u].reshape(dim_u, r)
                V_sol = params_sol[r*dim_u:r*(dim_u+dim_v)].reshape(dim_v, r)
                W_sol = params_sol[r*(dim_u+dim_v):].reshape(dim_w, r)
                
                # Check residual
                if res.cost < 1e-10:
                    is_valid, U_r, V_r, W_r = verify_solution(U_sol, V_sol, W_sol)
                    if is_valid:
                        best_solution = (U_r, V_r, W_r)
                        best_rank = r
                        found = True
                        break
            except Exception:
                continue
                
        if found:
            # If we found a solution for rank r, we can stop if we are going down
            # But we might find an even lower rank? Unlikely if 29 is the bound.
            # For now, break to save time for verification/polish or just return.
            break

    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
