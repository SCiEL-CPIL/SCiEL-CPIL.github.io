import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target Tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Function to compute error for given U, V, W
    def error(params):
        # params is a flat array of size R * (dim_U + dim_V + dim_W)
        R = len(params) // (dim_U + dim_V + dim_W)
        U = params[0:R*dim_U].reshape(R, dim_U)
        V = params[R*dim_U:R*(dim_U+dim_V)].reshape(R, dim_V)
        W = params[R*(dim_U+dim_V):].reshape(R, dim_W)
        
        # Compute sum of outer products
        # Efficient way: np.einsum('ri,rj,rk->ijk', U, V, W)
        T_recon = np.einsum('ri,rj,rk->ijk', U, V, W)
        diff = T_recon - T
        return np.sum(diff**2)

    # Try ranks from 25 upwards
    best_rank = 30
    best_params = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    for rank in [25, 26, 27, 28, 29, 30]:
        if time.time() > deadline:
            break
            
        success = False
        # Multiple restarts for this rank
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize parameters
            # Using random initialization
            init_size = rank * (dim_U + dim_V + dim_W)
            params = np.random.randn(init_size) * 0.1
            
            # Optimize
            # Using L-BFGS-B or similar
            # Since we need exact solution, we minimize error to 0
            res = minimize(error, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < 1e-9:
                # Found a solution
                success = True
                best_rank = rank
                best_params = res.x
                break
        
        if success:
            break

    if best_params is None:
        # Fallback to schoolbook if search fails
        best_rank = 30
        # Construct schoolbook
        U = np.zeros((dim_U, best_rank))
        V = np.zeros((dim_V, best_rank))
        W = np.zeros((dim_W, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Extract U, V, W from best_params
    R = best_rank
    U = best_params[0:R*dim_U].reshape(R, dim_U).T
    V = best_params[R*dim_U:R*(dim_U+dim_V)].reshape(R, dim_V).T
    W = best_params[R*(dim_U+dim_V):].reshape(R, dim_W).T
    
    # Snap to nearest half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify exactness
    T_recon = np.einsum('ri,rj,rk->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # If snapping failed, try rounding to integer? Or just return the float solution?
        # The task requires integer/half-integer.
        # If it's not exact, we should probably fall back to schoolbook to ensure validity,
        # or try a different restart.
        # For safety, if the snapped version is not exact, we might have a problem.
        # However, if the error was < 1e-9, the solution is very close to exact.
        # Given the structure of such problems, small integer solutions are likely.
        # If allclose fails, it means the solution wasn't close enough to a half-integer grid.
        # We will return the schoolbook solution to guarantee validity, even if rank is higher.
        # But wait, if we found a rank 25 real solution, it's likely there is a rational one.
        # Let's try to force integer/half-integer by checking a few nearby grid points?
        # Too expensive.
        
        # Fallback: Return Schoolbook
        best_rank = 30
        U = np.zeros((dim_U, best_rank))
        V = np.zeros((dim_V, best_rank))
        W = np.zeros((dim_W, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
