import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T
    # T[a, b, c] = 1 if a,b,c are consistent, else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Flattened target
    T_flat = T.flatten()
    
    def unpack(params):
        u_len = dim_U
        v_len = dim_V
        w_len = dim_W
        # params is concatenated U, V, W
        # U is (dim_U, R)
        # V is (dim_V, R)
        # W is (dim_W, R)
        # We need R to be known.
        # This function is called inside minimize, so R is fixed in closure.
        R = params.size // (u_len + v_len + w_len)
        u_flat = params[:u_len * R]
        v_flat = params[u_len * R : u_len * R + v_len * R]
        w_flat = params[u_len * R + v_len * R :]
        
        U = u_flat.reshape((u_len, R))
        V = v_flat.reshape((v_len, R))
        W = w_flat.reshape((w_len, R))
        return U, V, W

    def objective(params):
        U, V, W = unpack(params)
        # Compute S = sum_r U[:, r] V[:, r]^T ... wait.
        # S[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor contraction.
        # We can compute this efficiently.
        # S = np.einsum('ar,br,cr->abc', U, V, W)
        S = np.einsum('ar,br,cr->abc', U, V, W)
        residual = S.flatten() - T_flat
        return np.dot(residual, residual)

    def gradient(params):
        # Gradient of 0.5 * sum(residual^2) ? No, objective is sum(residual^2).
        # d/dp sum(S-T)^2 = 2 * (S-T) * dS/dp
        U, V, W = unpack(params)
        S = np.einsum('ar,br,cr->abc', U, V, W)
        residual = S - T
        # dS/du_ar = V[br] W[cr]
        # grad_U[a, r] = sum_{b,c} residual[a,b,c] V[b,r] W[c,r]
        # grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        # grad_V[b, r] = sum_{a,c} residual[a,b,c] U[a,r] W[c,r]
        # grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        # grad_W[c, r] = sum_{a,b} residual[a,b,c] U[a,r] V[b,r]
        # grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_rank = n * m * p
    best_solution = None
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin

    # Try ranks from 20 down to 1
    # Start with 20 as it's a likely candidate
    for R in range(20, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this R
        found = False
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            # Scale factors: U ~ 1, V ~ 1, W ~ 1
            # To avoid scaling issues, we can normalize W after each step?
            # Or just let the optimizer handle it.
            
            np.random.seed(restart)
            U_init = np.random.randn(dim_U, R)
            V_init = np.random.randn(dim_V, R)
            W_init = np.random.randn(dim_W, R)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimization
            try:
                res = minimize(objective, params_init, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.success or res.fun < 1e-9:
                    # Verify
                    U, V, W = unpack(res.x)
                    S = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(S - T))
                    if err < 1e-8:
                        # Found a valid solution
                        best_rank = R
                        best_solution = (U, V, W)
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break

    if best_solution:
        U, V, W = best_solution
        # Convert to lists of lists as required
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
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
