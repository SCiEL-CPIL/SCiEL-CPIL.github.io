import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Define the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j1, j2} ?
    # No, the condition is:
    # sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1, j2}
    
    def compute_error(params, rank):
        # params contains U, V, W flattened
        # This is inefficient, we should pass U, V, W directly or use a different structure
        pass

    # Efficient error calculation for given U, V, W
    def error_vec(U, V, W, rank):
        errors = []
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        errors.append(actual - expected)
        return np.array(errors)

    # Try to find rank 62
    target_rank = rank_school - 1
    if target_rank < best_rank:
        # Random initialization
        # We can start with schoolbook and remove one column, then perturb
        # Or just random
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            # Initialize U, V, W
            # Start with schoolbook, drop last column, and add small noise
            U = U_school[:, :target_rank].copy()
            V = V_school[:, :target_rank].copy()
            W = W_school[:, :target_rank].copy()
            
            # Add noise to help escape local minima
            U += np.random.randn(n * m, target_rank) * 0.1
            V += np.random.randn(m * p, target_rank) * 0.1
            W += np.random.randn(p * n, target_rank) * 0.1
            
            # Flatten for optimizer
            def func(params):
                U_flat = params[:n * m * target_rank]
                V_flat = params[n * m * target_rank : n * m * target_rank + m * p * target_rank]
                W_flat = params[n * m * target_rank + m * p * target_rank :]
                
                U_cur = U_flat.reshape(n * m, target_rank)
                V_cur = V_flat.reshape(m * p, target_rank)
                W_cur = W_flat.reshape(p * n, target_rank)
                
                return error_vec(U_cur, V_cur, W_cur, target_rank)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(func, x0, method='lm', max_nfev=1000)
                if result.success or np.linalg.norm(result.fun) < 1e-8:
                    U_opt = result.x[:n * m * target_rank].reshape(n * m, target_rank)
                    V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                    W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                    
                    # Verify
                    err = error_vec(U_opt, V_opt, W_opt, target_rank)
                    if np.linalg.norm(err) < 1e-6:
                        best_rank = target_rank
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        break
            except Exception:
                continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
