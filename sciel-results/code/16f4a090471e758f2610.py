import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Target ranks to try, starting from lowest potential improvement
    # Schoolbook rank is 24.
    # Try 19, 20, 21, 22, 23
    target_ranks = [19, 20, 21, 22, 23]
    
    # Pre-allocate indices for faster error computation
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Reshape to 3D: U_3d (n, m, R), V_3d (m, p, R), W_3d (p, n, R)
    
    def compute_error(U, V, W):
        # U, V, W are 2D arrays of shape (rows, R)
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # M[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        M = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Error is sum of (M - 1)^2
        error = np.sum((M - 1.0)**2)
        return error

    def verify_solution(U, V, W):
        # Check if entries are close to half-integers
        # And if the decomposition is exact
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if rounding changed the values significantly
        # If the optimization found a solution very close to half-integers, 
        # the difference should be tiny.
        if np.max(np.abs(U - U_r)) > 1e-6 or np.max(np.abs(V - V_r)) > 1e-6 or np.max(np.abs(W - W_r)) > 1e-6:
            return None
            
        # Verify the rounded solution
        error = compute_error(U_r, V_r, W_r)
        if error < 1e-8:
            return U_r, V_r, W_r
        return None

    def try_rank(R):
        if time.time() > deadline:
            return None
            
        # Number of variables: R * (n*m + m*p + p*n)
        # 19 * (6 + 12 + 8) = 19 * 26 = 494 variables.
        
        # Try multiple seeds
        # We limit the number of iterations per rank based on time
        max_iter = 10
        
        for seed in range(max_iter):
            if time.time() > deadline:
                return None
                
            np.random.seed(seed)
            
            # Initialize
            # Try different initialization strategies
            if seed % 2 == 0:
                # Random small values
                U = np.random.randn(n * m, R) * 0.5
                V = np.random.randn(m * p, R) * 0.5
                W = np.random.randn(p * n, R) * 0.5
            else:
                # Random discrete values -1, 0, 1
                U = np.random.choice([-1, 0, 1], size=(n * m, R)).astype(float)
                V = np.random.choice([-1, 0, 1], size=(m * p, R)).astype(float)
                W = np.random.choice([-1, 0, 1], size=(p * n, R)).astype(float)
                
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(p):
                U_flat = p[:n*m*R]
                V_flat = p[n*m*R : n*m*R + m*p*R]
                W_flat = p[n*m*R + m*p*R :]
                
                U = U_flat.reshape(n*m, R)
                V = V_flat.reshape(m*p, R)
                W = W_flat.reshape(p*n, R)
                
                return compute_error(U, V, W)
            
            # Optimize
            # L-BFGS-B is good for large scale, but BFGS might be better for small?
            # Let's use L-BFGS-B with maxiter limit
            try:
                result = minimize(
                    objective,
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
            except Exception as e:
                continue
                
            if result.fun < 1e-6:
                # Extract solution
                U_sol = result.x[:n*m*R].reshape(n*m, R)
                V_sol = result.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_sol = result.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                sol = verify_solution(U_sol, V_sol, W_sol)
                if sol:
                    return sol

        return None

    # Try ranks
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        sol = try_rank(R)
        if sol:
            U, V, W = sol
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
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
