import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U shape: (n*m, rank) -> (6, R)
    # V shape: (m*p, rank) -> (9, R)
    # W shape: (p*n, rank) -> (6, R)
    
    # Precompute indices for the tensor T
    # T_{(i,j), (j,k), (k,i)} = 1
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                indices.append((I, J, K))
                
    indices = np.array(indices)
    
    def objective(params):
        R = len(params) // 21
        # Reshape params
        # U: 6*R, V: 9*R, W: 6*R
        U_flat = params[:6*R]
        V_flat = params[6*R:6*R+9*R]
        W_flat = params[6*R+9*R:]
        
        U = U_flat.reshape(6, R)
        V = V_flat.reshape(9, R)
        W = W_flat.reshape(6, R)
        
        # Compute sum_{r} U[I,r] V[J,r] W[K,r]
        # We need to evaluate at the specific indices
        U_I = U[indices[:, 0], :]
        V_J = V[indices[:, 1], :]
        W_K = W[indices[:, 2], :]
        
        prod = U_I * V_J * W_K
        sums = np.sum(prod, axis=1)
        
        err = sums - 1.0
        return np.sum(err**2)

    def check_solution(U, V, W):
        # Check if entries are close to half-integers
        # And if the decomposition is exact
        U_check = np.round(U * 2) / 2
        V_check = np.round(V * 2) / 2
        W_check = np.round(W * 2) / 2
        
        # Check if rounded matrices are close to original
        if not (np.allclose(U, U_check, atol=1e-4) and 
                np.allclose(V, V_check, atol=1e-4) and 
                np.allclose(W, W_check, atol=1e-4)):
            return False, None, None, None
            
        # Verify exact decomposition with rounded values
        U_I = U_check[indices[:, 0], :]
        V_J = V_check[indices[:, 1], :]
        W_K = W_check[indices[:, 2], :]
        
        prod = U_I * V_J * W_K
        sums = np.sum(prod, axis=1)
        
        if np.allclose(sums, 1.0, atol=1e-9):
            return True, U_check, V_check, W_check
        else:
            return False, None, None, None

    start_time = time.time()
    time_limit = 110 # Leave some buffer
    
    # Try ranks from 12 up to 17
    # If we find a valid solution, return it
    for R in range(12, 18):
        if time.time() - start_time > time_limit:
            break
            
        # Number of restarts
        max_restarts = 10
        restart_time = (time_limit - (time.time() - start_time)) / max_restarts
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U_init = np.random.randn(6, R)
            V_init = np.random.randn(9, R)
            W_init = np.random.randn(6, R)
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimization
            try:
                res = minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 100, 'maxfun': 1000})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Potential solution found
                U_opt = res.x[:6*R].reshape(6, R)
                V_opt = res.x[6*R:6*R+9*R].reshape(9, R)
                W_opt = res.x[6*R+9*R:].reshape(6, R)
                
                valid, U_final, V_final, W_final = check_solution(U_opt, V_opt, W_opt)
                if valid:
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # Fallback to trivial rank 18
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
