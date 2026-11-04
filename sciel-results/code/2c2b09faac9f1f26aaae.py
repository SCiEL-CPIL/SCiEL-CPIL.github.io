"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = DIMENSIONS
    # Build the target tensor T with shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    start_time = time.time()
    max_time = 110
    
    # Start from schoolbook and try to reduce rank
    schoolbook_rank = n * m * p
    U_school = np.zeros((dim_U, schoolbook_rank))
    V_school = np.zeros((dim_V, schoolbook_rank))
    W_school = np.zeros((dim_W, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank by starting from subsets of schoolbook columns
    for rank in range(1, schoolbook_rank + 1):
        if time.time() - start_time > max_time:
            break
        
        # Try different subsets of columns from schoolbook
        for subset_trial in range(10):
            if time.time() - start_time > max_time:
                break
            
            if rank == schoolbook_rank:
                # Just use schoolbook
                U_init = U_school.copy()
                V_init = V_school.copy()
                W_init = W_school.copy()
            else:
                # Randomly select 'rank' columns from schoolbook and use as initial guess
                # But this may not be a good initialization
                np.random.seed(subset_trial)
                selected = np.random.choice(schoolbook_rank, rank, replace=False)
                U_init = U_school[:, selected].copy()
                V_init = V_school[:, selected].copy()
                W_init = W_school[:, selected].copy()
                # Add small random perturbation
                U_init += np.random.randn(dim_U, rank) * 0.01
                V_init += np.random.randn(dim_V, rank) * 0.01
                W_init += np.random.randn(dim_W, rank) * 0.01
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(x):
                U = x[:dim_U * rank].reshape(dim_U, rank)
                V = x[dim_U * rank:dim_U * rank + dim_V * rank].reshape(dim_V, rank)
                W = x[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                return (recon - T).flatten()
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=3000)
                if np.max(np.abs(result.fun)) < 1e-8:
                    U = result.x[:dim_U * rank].reshape(dim_U, rank)
                    V = result.x[dim_U * rank:dim_U * rank + dim_V * rank].reshape(dim_V, rank)
                    W = result.x[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
                    
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(recon, T, atol=1e-8):
                        U_list = [[float(x) for x in row] for row in U]
                        V_list = [[float(x) for x in row] for row in V]
                        W_list = [[float(x) for x in row] for row in W]
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
    
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
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
