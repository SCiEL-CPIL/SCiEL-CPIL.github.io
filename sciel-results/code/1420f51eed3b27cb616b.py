import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Standard rank
    rank_std = n * m * p
    U_std = [[0] * rank_std for _ in range(n * m)]
    V_std = [[0] * rank_std for _ in range(m * p)]
    W_std = [[0] * rank_std for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j][col] = 1
                V_std[j * p + k][col] = 1
                W_std[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    # Try rank 17
    target_rank = 17
    if target_rank < rank_std:
        start_time = time.time()
        deadline = 110 # Safety margin
        
        # Residual function for least_squares
        def residual(params):
            # params: flattened U, V, W
            # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            
            # T_rec = sum_t U[:,t] outer V[:,t] outer W[:,t]
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            return (T_flat - T_rec.flatten())
            
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(restart)
            # Scale initialization to help convergence
            params = np.random.randn(dim_u*target_rank + dim_v*target_rank + dim_w*target_rank) * 0.5
            
            res = least_squares(residual, params, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if res.success or res.cost < 1e-10:
                # Check if entries are close to integer/half-integer
                U_sol = res.x[0:dim_u*target_rank].reshape((dim_u, target_rank))
                V_sol = res.x[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
                W_sol = res.x[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
                
                # Check 2*entries are close to integers
                is_half_int = True
                for mat in [U_sol, V_sol, W_sol]:
                    doubled = 2.0 * mat
                    rounded = np.round(doubled)
                    if np.max(np.abs(doubled - rounded)) > 1e-4:
                        is_half_int = False
                        break
                
                if is_half_int:
                    # Construct the result with rounded values
                    U_res = np.round(2.0 * U_sol) / 2.0
                    V_res = np.round(2.0 * V_sol) / 2.0
                    W_res = np.round(2.0 * W_sol) / 2.0
                    
                    # Verify exactness
                    T_check = np.einsum('at,bt,ct->abc', U_res, V_res, W_res)
                    if np.allclose(T_check, T, atol=1e-9):
                        # Convert to list of lists
                        U_list = U_res.tolist()
                        V_list = V_res.tolist()
                        W_list = W_res.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
    # Return standard rank 18
    return {"U": U_std, "V": V_std, "W": W_std}
# EVOLVE-BLOCK-END
