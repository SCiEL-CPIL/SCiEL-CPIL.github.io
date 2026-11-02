import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # Try ranks from 10 to 12
    for rank in range(10, 13):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank):
    """Try to find a rank decomposition using scipy least_squares."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    num_eq = n * m * p
    total_vars = dim_u * rank + dim_v * rank + dim_w * rank
    
    np.random.seed(42)
    
    for attempt in range(20):
        # Random initialization with varying scales
        scale = 0.3 + 0.1 * attempt
        x0 = np.random.randn(total_vars) * scale
        
        def residuals(x):
            U = x[:dim_u*rank].reshape(dim_u, rank)
            V = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            res = np.zeros(num_eq)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[idx] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1
                        idx += 1
            return res
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=5000)
            if np.max(np.abs(sol.fun)) < 1e-9:
                U = sol.x[:dim_u*rank].reshape(dim_u, rank)
                V = sol.x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W = sol.x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                
                # Verify exactness
                max_res = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_res = max(max_res, abs(val - 1))
                
                if max_res < 1e-8:
                    U_list = [[float(U[i,c]) for c in range(rank)] for i in range(dim_u)]
                    V_list = [[float(V[i,c]) for c in range(rank)] for i in range(dim_v)]
                    W_list = [[float(W[i,c]) for c in range(rank)] for i in range(dim_w)]
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    return None
# EVOLVE-BLOCK-END
