import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 110
    
    # Build target tensor T
    # T[i, j, k] = 1 if (i,j) in A, (j,k) in B, result in C
    # A is n x m, B is m x p, C is n x p
    # T is a tensor of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    # Try rank 35
    rank = 35
    if time.time() < deadline - 5:
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(n*p, rank)
        
        def residuals(params):
            U_flat, V_flat, W_flat = np.split(params, [n*m*rank, n*m*rank + m*p*rank])
            U_r = U_flat.reshape(n*m, rank)
            V_r = V_flat.reshape(m*p, rank)
            W_r = W_flat.reshape(n*p, rank)
            
            # Compute decomposition
            # T_decomp[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is slow to compute directly.
            # Use einsum
            T_decomp = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            return (T - T_decomp).flatten()
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try a few random restarts
        best_sol = None
        best_err = np.inf
        for _ in range(5):
            if time.time() > deadline - 2: break
            x0 = np.random.randn(len(x0)) * 0.1
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if sol.cost < best_err:
                    best_err = sol.cost
                    best_sol = sol
            except Exception:
                pass
                
        if best_sol is not None and best_err < 1e-8:
            U_flat, V_flat, W_flat = np.split(best_sol.x, [n*m*rank, n*m*rank + m*p*rank])
            U_out = U_flat.reshape(n*m, rank).tolist()
            V_out = V_flat.reshape(m*p, rank).tolist()
            W_out = W_flat.reshape(n*p, rank).tolist()
            return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(n * p)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
