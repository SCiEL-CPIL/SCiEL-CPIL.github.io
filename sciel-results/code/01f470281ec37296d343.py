import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Target ranks to try
    target_ranks = [20, 21, 22, 23, 24, 25, 26]
    
    start_time = time.time()
    deadline = start_time + 140 # Leave 10s margin
    
    def tensor_from_decomp(U, V, W):
        # U: 9 x R, V: 9 x R, W: 9 x R
        # Returns tensor of shape (3, 3, 3)
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Note indices:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        T = np.zeros((n, m, p))
        R = U.shape[1]
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        return T

    def get_true_tensor():
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T

    T_true = get_true_tensor()
    T_flat_true = T_true.flatten()

    def objective(params):
        R = len(target_ranks[0]) # No, R is passed via closure or we restructure
        # This function needs to know R. We will define it inside the loop.
        pass

    best_result = None
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Define objective for this R
        def obj_fn(params):
            U = params[:9*R].reshape(9, R)
            V = params[9*R:18*R].reshape(9, R)
            W = params[18*R:27*R].reshape(9, R)
            T = tensor_from_decomp(U, V, W)
            diff = T.flatten() - T_flat_true
            return np.dot(diff, diff)
        
        def grad_fn(params):
            U = params[:9*R].reshape(9, R)
            V = params[9*R:18*R].reshape(9, R)
            W = params[18*R:27*R].reshape(9, R)
            T = tensor_from_decomp(U, V, W)
            diff = T.flatten() - T_flat_true
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(R):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dT = diff[i, j, k]
                            # dT/dU[i*m+j, r] = v[j*p+k, r] * w[k*n+i, r]
                            grad_U[i * m + j, r] += dT * v[j * p + k, r] * w[k * n + i, r]
                            grad_V[j * p + k, r] += dT * u[i * m + j, r] * w[k * n + i, r]
                            grad_W[k * n + i, r] += dT * u[i * m + j, r] * v[j * p + k, r]
                            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        # Try a few restarts for this rank
        max_restarts = 10
        time_per_restart = 5.0
        restart_deadline = start_time + (deadline - start_time) / (len(target_ranks) * max_restarts) * 2 # Rough allocation
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale to avoid large numbers
            U_init = np.random.randn(9, R) * 0.5
            V_init = np.random.randn(9, R) * 0.5
            W_init = np.random.randn(9, R) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(obj_fn, params_init, method='L-BFGS-B', jac=grad_fn, 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 50})
                
                if res.success or res.fun < 1e-6:
                    # Check if close to integer/half-integer
                    U_opt = res.x[:9*R].reshape(9, R)
                    V_opt = res.x[9*R:18*R].reshape(9, R)
                    W_opt = res.x[18*R:27*R].reshape(9, R)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify
                    T_round = tensor_from_decomp(U_round, V_round, W_round)
                    if np.allclose(T_round, T_true, atol=1e-9):
                        # Convert to lists
                        U_list = U_round.tolist()
                        V_list = V_round.tolist()
                        W_list = W_round.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                pass

    return {"U": U_school, "V": V_school, "W": W_school}
