import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute indices for speed
    # Not strictly necessary with the loop structure
    
    for rank in range(25, 36):
        if time.time() > deadline:
            break
            
        best_error = float('inf')
        best_params = None
        
        num_tries = 50
        for t in range(num_tries):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use small values to help convergence
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*rank], x[n*m*rank:n*m*rank+m*p*rank], x[n*m*rank+m*p*rank:]
                U_mat = U_flat.reshape(n * m, rank)
                V_mat = V_flat.reshape(m * p, rank)
                W_mat = W_flat.reshape(p * n, rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        u_row = U_mat[i * m + j]
                        for k in range(p):
                            v_row = V_mat[j * p + k]
                            w_row = W_mat[k * n + i]
                            term = np.dot(u_row, v_row * w_row)
                            error += (1.0 - term) ** 2
                return error
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-8})
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_error < 1e-6 and best_params is not None:
            U_flat, V_flat, W_flat = best_params[:n*m*rank], best_params[n*m*rank:n*m*rank+m*p*rank], best_params[n*m*rank+m*p*rank:]
            U_mat = U_flat.reshape(n * m, rank)
            V_mat = V_flat.reshape(m * p, rank)
            W_mat = W_flat.reshape(p * n, rank)
            
            # Try rounding to half-integers
            U_rounded = np.round(U_mat * 2) / 2
            V_rounded = np.round(V_mat * 2) / 2
            W_rounded = np.round(W_mat * 2) / 2
            
            error_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term = np.dot(U_rounded[i * m + j], V_rounded[j * p + k] * W_rounded[k * n + i])
                        error_rounded += (1.0 - term) ** 2
                        
            if error_rounded < 1e-6:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            else:
                # If rounding fails significantly, return floats
                return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

    # Fallback
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
