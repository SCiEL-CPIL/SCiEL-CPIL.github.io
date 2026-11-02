import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Target rank to try
    target_rank = 12
    
    # Function to compute the error for given U, V, W
    def compute_error(U, V, W):
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # T[i, j, k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # We want T[i, j, k] = 1 for all i, j, k.
        # Actually, the tensor M has 1s at (i, j, k) for the multiplication.
        # M[i*m+j, j*p+k, k*n+i] = 1.
        # So we want sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1.
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    # Use vectorized dot product for speed
                    val = np.dot(U[i*m+j], V[j*p+k])
                    # Wait, W is also involved.
                    # val = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    # This is element-wise product then sum.
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    error += (val - 1.0)**2
        return error

    # Function to flatten params
    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    # Function to unflatten params
    def unflatten(params):
        U_size = n * m * target_rank
        V_size = m * p * target_rank
        W_size = p * n * target_rank
        U = params[:U_size].reshape(n*m, target_rank)
        V = params[U_size:U_size+V_size].reshape(m*p, target_rank)
        W = params[U_size+V_size:].reshape(p*n, target_rank)
        return U, V, W

    best_params = None
    best_loss = float('inf')
    
    # Try multiple random initializations
    for seed in range(50):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        params = flatten(U, V, W)
        
        # Optimize
        # Use L-BFGS-B as it is robust
        result = minimize(
            lambda p: compute_error(*unflatten(p)),
            params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
            
            # Check if we found a very good solution
            if best_loss < 1e-8:
                break

    if best_params is not None and best_loss < 1e-6:
        U_sol, V_sol, W_sol = unflatten(best_params)
        
        # Round to nearest 0.5
        U_rounded = np.round(U_sol * 2) / 2
        V_rounded = np.round(V_sol * 2) / 2
        W_rounded = np.round(W_sol * 2) / 2
        
        # Verify rounded solution
        error = compute_error(U_rounded, V_rounded, W_rounded)
        
        if error < 1e-6:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }

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
