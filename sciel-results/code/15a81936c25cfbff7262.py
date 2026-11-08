import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Precompute indices for efficiency
    # Residuals are for all i, k, j1, j2
    # Target is 1 if j1 == j2 else 0
    
    # We will try ranks from low to high
    # Lower bound approx 30
    # Upper bound 120
    
    for R in range(30, 121):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this R
        # Number of restarts decreases as R increases or time runs out
        max_restarts = 5 if R < 50 else 2
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize variables
            # U: n*m x R
            # V: m*p x R
            # W: p*n x R
            size_u = n * m
            size_v = m * p
            size_w = p * n
            total_vars = R * (size_u + size_v + size_w)
            
            # Random initialization
            x0 = np.random.uniform(-1, 1, total_vars)
            
            # Residual function
            def residuals(x):
                U = x[:R*size_u].reshape(R, size_u).T # n*m x R
                V = x[R*size_u:R*(size_u+size_v)].reshape(R, size_v).T # m*p x R
                W = x[R*(size_u+size_v):].reshape(R, size_w).T # p*n x R
                
                res = np.zeros((n, p, m, m))
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                # Sum over r
                                # U[i*m+j1, r]
                                # V[j2*p+k, r]
                                # W[k*n+i, r]
                                u_col = U[i*m+j1, :]
                                v_col = V[j2*p+k, :]
                                w_col = W[k*n+i, :]
                                val = np.dot(u_col, v_col * w_col)
                                target = 1.0 if j1 == j2 else 0.0
                                res[i, k, j1, j2] = val - target
                return res.ravel()
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if sol.success and sol.cost < 1e-8:
                    # Extract U, V, W
                    U = sol.x[:R*size_u].reshape(R, size_u).T
                    V = sol.x[R*size_u:R*(size_u+size_v)].reshape(R, size_v).T
                    W = sol.x[R*(size_u+size_v):].reshape(R, size_w).T
                    
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check error of rounded solution
                    # We can reuse the residual function logic or just check a few points
                    # Let's check the max error
                    max_err = 0
                    for i in range(n):
                        for k in range(p):
                            for j1 in range(m):
                                for j2 in range(m):
                                    u_col = U_rounded[i*m+j1, :]
                                    v_col = V_rounded[j2*p+k, :]
                                    w_col = W_rounded[k*n+i, :]
                                    val = np.dot(u_col, v_col * w_col)
                                    target = 1.0 if j1 == j2 else 0.0
                                    err = abs(val - target)
                                    if err > max_err:
                                        max_err = err
                                        
                    if max_err < 1e-6:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
            except Exception:
                pass
                
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
