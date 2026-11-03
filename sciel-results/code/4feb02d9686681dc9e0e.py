import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    total_time_limit = 110.0
    start_time = time.time()
    
    def get_rank_55_solution():
        # ... (same as above) ...
        # I will copy the rank 55 generation code here.
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W

    U, V, W = get_rank_55_solution()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 55

    def check_validity(U, V, W, tol=1e-6):
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        err = 0.0
        for i in range(n):
            for k in range(p):
                W_k_i = W_mat[k, i, :]
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                A = U_i * W_k_i[None, :]
                M_ik = A.T @ V_k
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err < tol

    # Try to find rank 54
    # We will try removing each of the 55 columns and see if we can fit the rest.
    # This is a non-convex problem.
    # We will use a simple gradient descent on the flattened parameters.
    
    def objective(params):
        R = 54
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R :].reshape(p*n, R)
        
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        err = 0.0
        for i in range(n):
            for k in range(p):
                W_k_i = W_mat[k, i, :]
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                A = U_i * W_k_i[None, :]
                M_ik = A.T @ V_k
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    # Try a few random restarts for rank 54
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() - start_time > total_time_limit - 10:
            break
        
        # Initialize with rank 55, drop a random column
        drop_col = random.randint(0, 54)
        U_init = np.delete(U, drop_col, axis=1)
        V_init = np.delete(V, drop_col, axis=1)
        W_init = np.delete(W, drop_col, axis=1)
        
        # Add small noise
        U_init += np.random.normal(0, 0.1, U_init.shape)
        V_init += np.random.normal(0, 0.1, V_init.shape)
        W_init += np.random.normal(0, 0.1, W_init.shape)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Run minimization
        # Using L-BFGS-B or similar
        # This might be slow. Let's limit iterations.
        try:
            res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 1000})
            if res.fun < 1e-6:
                # Check if it's close to integers
                # If so, we found a rank 54 solution
                params_best = res.x
                U_new = params_best[:n*m*54].reshape(n*m, 54)
                V_new = params_best[n*m*54 : n*m*54 + m*p*54].reshape(m*p, 54)
                W_new = params_best[n*m*54 + m*p*54 :].reshape(p*n, 54)
                
                # Snap to nearest 0.5
                U_new = np.round(U_new * 2) / 2
                V_new = np.round(V_new * 2) / 2
                W_new = np.round(W_new * 2) / 2
                
                if check_validity(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = 54
                    break
        except Exception:
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
