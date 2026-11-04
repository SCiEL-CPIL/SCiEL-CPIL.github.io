"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Define the residual function for a given rank
    def make_residuals(target_rank):
        def residuals(params):
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            res = np.empty(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        res[idx] = val - 1
                        idx += 1
            return res
        return residuals
    
    # Try to find a lower-rank decomposition
    # Start from rank 140 and go down
    for target_rank in range(140, 10):
        if time.time() > deadline - 30:
            break
        
        # Try a few initializations
        found = False
        for seed in [42, 123, 456]:
            if time.time() > deadline - 30:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            residuals = make_residuals(target_rank)
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=200)
                
                if result.success and np.max(np.abs(result.fun)) < 1e-8:
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    best_rank = target_rank
                    best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END

