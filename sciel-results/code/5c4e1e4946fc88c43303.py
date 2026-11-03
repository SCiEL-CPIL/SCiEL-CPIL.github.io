import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # We aim to decompose 2x3x6 by splitting into two 2x3x3 blocks.
    # Total rank = 2 * rank(2x3x3)
    # Target rank for 2x3x3 is 16 (total 32) or 15 (total 30).
    # Schoolbook is 18 (total 36).
    
    n, m, p = 2, 3, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    best_solution = None
    best_rank = 18 # Schoolbook for 2x3x3
    
    # Try to find a lower rank decomposition
    for R in [16, 17]:
        if time.time() > deadline - 10:
            break
        
        found = False
        # Use multiple restarts
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values, scaled
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            def objective(params):
                U_flat = params[:n*m*R].reshape(n*m, R)
                V_flat = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = params[n*m*R + m*p*R:].reshape(p*n, R)
                T_sum = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
                residual = T_sum - T
                return np.sum(residual**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < 1e-6:
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        best_solution = (U_rounded, V_rounded, W_rounded, R)
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
            
    if best_solution:
        U_233, V_233, W_233, R_233 = best_solution
        R_236 = 2 * R_233
        n, m, p = 2, 3, 6
        
        U_236 = np.zeros((n*m, R_236))
        V_236 = np.zeros((m*p, R_236))
        W_236 = np.zeros((p*n, R_236))
        
        # First half (columns 0..R_233-1) correspond to B[:, 0:3]
        U_236[:, :R_233] = U_233
        V_236[:m*3, :R_233] = V_233
        W_236[:3*n, :R_233] = W_233
        
        # Second half (columns R_233..2*R_233-1) correspond to B[:, 3:6]
        U_236[:, R_233:] = U_233
        V_236[m*3:, R_233:] = V_233
        W_236[3*n:, R_233:] = W_233
        
        return {
            "U": U_236.tolist(),
            "V": V_236.tolist(),
            "W": W_236.tolist()
        }
    else:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
