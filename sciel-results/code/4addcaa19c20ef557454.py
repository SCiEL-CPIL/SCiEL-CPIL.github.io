import numpy as np
import time
from scipy.optimize import differential_evolution

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank
    target_rank = 21
    
    # Shapes
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Total parameters
    num_params = (size_U + size_V + size_W) * target_rank
    
    # Generate random test matrices
    # Use a fixed seed for reproducibility in the search logic
    np.random.seed(42)
    num_tests = 5
    A_tests = [np.random.randn(n, m) for _ in range(num_tests)]
    B_tests = [np.random.randn(m, p) for _ in range(num_tests)]
    
    def objective(params):
        # Reshape params
        U = params[:size_U * target_rank].reshape(size_U, target_rank)
        V = params[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape(size_V, target_rank)
        W = params[size_U * target_rank + size_V * target_rank :].reshape(size_W, target_rank)
        
        err = 0.0
        for idx in range(num_tests):
            A = A_tests[idx]
            B = B_tests[idx]
            C_target = A @ B
            
            # Compute C from U, V, W
            # C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_calc = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    w_vec = W[k * n + i, :]
                    for j in range(m):
                        u_vec = U[i * m + j, :]
                        v_vec = V[j * p + k, :]
                        C_calc[i, k] += np.sum(u_vec * v_vec * w_vec)
            
            err += np.sum((C_calc - C_target) ** 2)
        
        return err

    # Bounds for differential evolution
    # Entries are likely small integers or half-integers.
    # Let's search in [-5, 5]
    bounds = [(-5.0, 5.0)] * num_params
    
    start_time = time.time()
    timeout = 100  # Leave 20s for verification and fallback
    
    best_solution = None
    best_score = float('inf')
    
    try:
        # differential_evolution is slow but robust.
        # We use a small population and limited iterations to fit in time.
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6, 
            update_interval=10,
            polish=False, # Disable polish to save time, we will round manually
            seed=42,
            init='sobol'
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_solution = result.x
            
    except Exception:
        pass

    # Check if we found a good solution
    if best_solution is not None and best_score < 1e-3:
        # Round to nearest 0.5
        U_flat = best_solution[:size_U * target_rank]
        V_flat = best_solution[size_U * target_rank : size_U * target_rank + size_V * target_rank]
        W_flat = best_solution[size_U * target_rank + size_V * target_rank :]
        
        U_rounded = np.round(U_flat * 2) / 2
        V_rounded = np.round(V_flat * 2) / 2
        W_rounded = np.round(W_flat * 2) / 2
        
        # Verify rounded solution with more tests
        np.random.seed(123)
        num_verify_tests = 10
        A_verify = [np.random.randn(n, m) for _ in range(num_verify_tests)]
        B_verify = [np.random.randn(m, p) for _ in range(num_verify_tests)]
        
        err = 0.0
        for idx in range(num_verify_tests):
            A = A_verify[idx]
            B = B_verify[idx]
            C_target = A @ B
            C_calc = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    w_vec = W_rounded.reshape(size_W, target_rank)[k * n + i, :]
                    for j in range(m):
                        u_vec = U_rounded.reshape(size_U, target_rank)[i * m + j, :]
                        v_vec = V_rounded.reshape(size_V, target_rank)[j * p + k, :]
                        C_calc[i, k] += np.sum(u_vec * v_vec * w_vec)
            err += np.sum((C_calc - C_target) ** 2)
        
        if err < 1e-6:
            return {
                "U": U_rounded.reshape(size_U, target_rank).tolist(),
                "V": V_rounded.reshape(size_V, target_rank).tolist(),
                "W": W_rounded.reshape(size_W, target_rank).tolist()
            }

    # Fallback to seed (rank 24)
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
