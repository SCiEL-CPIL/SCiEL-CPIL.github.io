import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_std = n * m * p  # 18
    
    # Build standard rank-18 decomposition
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
    
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Try rank 17 first, then rank 16
    for target_rank in [17, 16]:
        if time.time() > deadline:
            break
            
        U_arr = np.array(U_std, dtype=float)
        V_arr = np.array(V_std, dtype=float)
        W_arr = np.array(W_std, dtype=float)
        
        def objective(params):
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            E = T_flat - T_rec.flatten()
            return np.dot(E, E)
        
        def gradient(params):
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            E_tensor = (T - T_rec)
            
            grad_U = -2 * np.einsum('abc,bt,ct->at', E_tensor, V, W)
            grad_V = -2 * np.einsum('abc,at,ct->bt', E_tensor, U, W)
            grad_W = -2 * np.einsum('abc,at,bt->ct', E_tensor, U, V)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        best_err = float('inf')
        best_params = None
        
        # Multiple restart strategies
        num_restarts = 20 if target_rank == 17 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 13 + 42)
            
            # Strategy 1: Near-standard initialization with perturbation
            if restart < num_restarts // 2:
                U_init = np.zeros((dim_u, target_rank))
                V_init = np.zeros((dim_v, target_rank))
                W_init = np.zeros((dim_w, target_rank))
                
                # Copy first target_rank-1 columns from standard
                for t in range(target_rank - 1):
                    U_init[:, t] = U_arr[:, t]
                    V_init[:, t] = V_arr[:, t]
                    W_init[:, t] = W_arr[:, t]
                
                # Last column: average of last two standard columns
                U_init[:, target_rank-1] = (U_arr[:, target_rank-1] + U_arr[:, target_rank]) / 2.0
                V_init[:, target_rank-1] = (V_arr[:, target_rank-1] + V_arr[:, target_rank]) / 2.0
                W_init[:, target_rank-1] = (W_arr[:, target_rank-1] + W_arr[:, target_rank]) / 2.0
                
                # Add perturbation
                scale = 0.1
                U_init += np.random.randn(dim_u, target_rank) * scale
                V_init += np.random.randn(dim_v, target_rank) * scale
                W_init += np.random.randn(dim_w, target_rank) * scale
            else:
                # Strategy 2: Random initialization with structure
                U_init = np.random.randn(dim_u, target_rank) * 0.5
                V_init = np.random.randn(dim_v, target_rank) * 0.5
                W_init = np.random.randn(dim_w, target_rank) * 0.5
                
                # Ensure some structure: make first few columns similar to standard
                for t in range(min(3, target_rank)):
                    U_init[:, t] += U_arr[:, t] * 0.5
                    V_init[:, t] += V_arr[:, t] * 0.5
                    W_init[:, t] += W_arr[:, t] * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            from scipy.optimize import minimize
            
            try:
                res = minimize(objective, params_init, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-15, 'disp': False})
                
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x.copy()
                
                # Check for exact solution
                if res.fun < 1e-10:
                    U_sol = res.x[0:dim_u*target_rank].reshape((dim_u, target_rank))
                    V_sol = res.x[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
                    W_sol = res.x[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(2.0 * U_sol) / 2.0
                    V_rounded = np.round(2.0 * V_sol) / 2.0
                    W_rounded = np.round(2.0 * W_sol) / 2.0
                    
                    T_check = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    
                    # Also try integer rounding
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    T_check_int = np.einsum('at,bt,ct->abc', U_int, V_int, W_int)
                    if np.allclose(T_check_int, T, atol=1e-9):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
            except Exception:
                pass
        
        # If we found a good solution for this rank, try to polish it
        if best_err < 1e-8 and best_params is not None:
            U_sol = best_params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V_sol = best_params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W_sol = best_params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            
            # Polish with tighter tolerance
            params_polish = best_params.copy()
            from scipy.optimize import minimize
            try:
                res = minimize(objective, params_polish, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-18, 'disp': False})
                
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x.copy()
                    U_sol = res.x[0:dim_u*target_rank].reshape((dim_u, target_rank))
                    V_sol = res.x[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
                    W_sol = res.x[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            except Exception:
                pass
            
            # Final check for exact half-integer solution
            U_rounded = np.round(2.0 * U_sol) / 2.0
            V_rounded = np.round(2.0 * V_sol) / 2.0
            W_rounded = np.round(2.0 * W_sol) / 2.0
            
            T_check = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_check, T, atol=1e-9):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    # Return standard rank 18
    return {"U": U_std, "V": V_std, "W": W_std}
