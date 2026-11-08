import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank
    target_rank = 17
    
    # Initialize random matrices
    np.random.seed(42)
    U = np.random.rand(n * m, target_rank)
    V = np.random.rand(m * p, target_rank)
    W = np.random.rand(p * n, target_rank)
    
    # Define the function to minimize
    def residual(params):
        U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, n*m*target_rank + m*p*target_rank])
        U_mat = U_flat.reshape(n * m, target_rank)
        V_mat = V_flat.reshape(m * p, target_rank)
        W_mat = W_flat.reshape(p * n, target_rank)
        
        # Compute tensor decomposition
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want this to equal 1 for all i, j, k
        # Residual is the difference
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(target_rank):
                        val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    errors.append(val - 1)
        return errors

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Try to solve
    result = least_squares(residual, x0, method='lm', max_nfev=10000)
    
    if result.success and np.max(np.abs(result.fun)) < 1e-5:
        U_sol = result.x[:n*m*target_rank].reshape(n * m, target_rank)
        V_sol = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
        W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p * n, target_rank)
        
        # Round to nearest half-integer
        U_int = np.round(U_sol * 2) / 2
        V_int = np.round(V_sol * 2) / 2
        W_int = np.round(W_sol * 2) / 2
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(target_rank):
                        val += U_int[i * m + j, c] * V_int[j * p + k, c] * W_int[k * n + i, c]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
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
# EVOLVE-BLOCK-END
