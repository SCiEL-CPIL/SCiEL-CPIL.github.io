import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i,j,k] = 1
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # U shape (9, R), V shape (9, R), W shape (9, R)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    def error_func(params, R):
        # params: flattened U, V, W
        U = params[:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:27*R].reshape(9, R)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err

    def get_solution(R, seed):
        np.random.seed(seed)
        # Initialize with small random values
        init = np.random.randn(27 * R) * 0.1
        
        # Define objective
        obj = lambda x: error_func(x, R)
        
        # Use minimize
        # L-BFGS-B is good for large scale
        res = minimize(obj, init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.fun < 1e-9:
            U = res.x[:9*R].reshape(9, R)
            V = res.x[9*R:18*R].reshape(9, R)
            W = res.x[18*R:27*R].reshape(9, R)
            # Convert to list of lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    start_time = time.time()
    deadline = start_time + 100 # 100s deadline to be safe
    
    # Try ranks 23, 24, 25, 26
    for R in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        # Try a few seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            sol = get_solution(R, seed)
            if sol is not None:
                return sol
                
    # Fallback to rank 27
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
