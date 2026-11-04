import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[a, b, c] where:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a, b, c] = 1 if indices match, 0 otherwise
    
    N = n * m
    M = m * p
    P = p * n
    
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 115.0 # Safety margin for 120s limit
    
    best_rank = N * M * P # Fallback (Schoolbook)
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute matricizations for fast ALS
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)
    
    def check_solution(U, V, W, tol=1e-9):
        # Verify if U, V, W contain only half-integers
        # And if they reconstruct T exactly (within tol)
        # The problem states "integer/half-integer entries"
        # We check if 2*U, 2*V, 2*W are close to integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-6):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-6):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-6):
            return False
            
        # Snap to exact half-integers for verification
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.allclose(T_calc, T, atol=tol), U_r, V_r, W_r

    def polish(U, V, W):
        # Try to refine U, V, W to be exactly half-integers while matching T
        # We define a vector x containing all entries of U, V, W
        # Objective: || T - Tensor(x) ||^2 + penalty for non-half-integers
        
        def func(x):
            u = x[:N*R].reshape(N, R)
            v = x[N*R:N*R+M*R].reshape(M, R)
            w = x[N*R+M*R:].reshape(P, R)
            
            # Tensor product
            t_calc = np.einsum('ar,br,cr->abc', u, v, w)
            residual = (t_calc - T).flatten()
            
            # Penalty: distance to nearest half-integer
            # half_int = round(2*x)/2
            # dist = x - half_int
            penalty_u = (u - np.round(u*2)/2).flatten()
            penalty_v = (v - np.round(v*2)/2).flatten()
            penalty_w = (w - np.round(w*2)/2).flatten()
            
            # We want the penalty to be 0, so we minimize the sum of squares of these distances
            # But we also need the tensor product to be correct.
            # If we minimize residual first, then snap, we might fail.
            # This joint optimization is tricky. 
            # Instead, let's just return the residual and rely on the fact that 
            # if we are very close to a half-integer solution, the residual will be small.
            # Then we snap and check.
            
            # Actually, a better strategy for polishing:
            # Minimize || T - Tensor(x) ||^2 
            # Then check if x is close to half-integers.
            # If yes, snap and verify.
            return residual

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use least_squares to minimize the tensor reconstruction error
        # This might push the solution to a "generic" low error point, 
        # but if a half-integer solution exists, it might be a local minimum 
        # or near one.
        
        try:
            result = least_squares(func, x0, method='lm', max_nfev=500)
            x_opt = result.x
            U_opt = x_opt[:N*R].reshape(N, R)
            V_opt = x_opt[N*R:N*R+M*R].reshape(M, R)
            W_opt = x_opt[N*R+M*R:].reshape(P, R)
            
            # Check if this optimized solution is close to half-integers
            valid, U_r, V_r, W_r = check_solution(U_opt, V_opt, W_opt)
            if valid:
                return True, U_r, V_r, W_r
            else:
                return False, None, None, None
        except:
            return False, None, None, None

    # Search ranks
    # 3x3x3 min rank is 23. 3x3x4 is likely 24, 25, 26...
    # We try 23 to 35.
    for R in range(23, 36):
        if time.time() - start_time > timeout:
            break
            
        # Try multiple seeds
        # The number of seeds depends on time remaining
        seeds_tried = 0
        max_seeds = 20 # Adjust based on time
        
        for seed in range(max_seeds):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + R * 100 + seed)
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # ALS Loop
            converged = False
            for it in range(50):
                if time.time() - start_time > timeout:
                    break
                    
                # Update W
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break
                
                # Update U
                T_a_vec = T.reshape(N, M * P)
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V
                T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check convergence
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_calc - T)
                
                if error < 1e-7:
                    # Try to polish and snap
                    valid, U_r, V_r, W_r = check_solution(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    
                    # If not valid, try polishing
                    valid, U_r, V_r, W_r = polish(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                        
            if converged:
                break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
