import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 13
    # Try to find a solution for rank 13
    # If it fails, try 14, 15, etc.
    
    def residual(flat_params, R):
        # Reshape params
        U = flat_params[0:R*n*m].reshape(R, n*m)
        V = flat_params[R*n*m:R*(n*m+m*p)].reshape(R, m*p)
        W = flat_params[R*(n*m+m*p):].reshape(R, p*n)
        
        # Compute tensor T[i,j,k] for the condition
        # Condition: sum_alpha W[k*n+i, alpha] U[i*m+j, alpha] V[j'*p+k, alpha] == delta[j, j']
        # We need to check for all i, k, j, j'
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0.0
                        for alpha in range(R):
                            val += W[k*n+i, alpha] * U[i*m+j, alpha] * V[jp*p+k, alpha]
                        target = 1.0 if j == jp else 0.0
                        res.append(val - target)
        return np.array(res)

    best_result = None
    best_rank = n*m*p # Schoolbook
    
    # Try ranks from 13 upwards
    for R in range(13, 18):
        x0 = np.random.randn(R * (n*m + m*p + p*n))
        # Add a few restarts
        found = False
        for restart in range(5):
            if restart == 0:
                x0 = np.random.randn(R * (n*m + m*p + p*n)) * 0.1
            else:
                x0 = np.random.randn(R * (n*m + m*p + p*n))
            
            try:
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=10000)
                if result.cost < 1e-8:
                    # Found a solution
                    # Round to nearest 0.5
                    U = result.x[0:R*n*m].reshape(R, n*m)
                    V = result.x[R*n*m:R*(n*m+m*p)].reshape(R, m*p)
                    W = result.x[R*(n*m+m*p):].reshape(R, p*n)
                    
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    res_check = residual(np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()]), R)
                    if np.max(np.abs(res_check)) < 1e-6:
                        best_result = {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                        best_rank = R
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    if best_result is None:
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
        best_result = {'U': U, 'V': V, 'W': W}

    return best_result
