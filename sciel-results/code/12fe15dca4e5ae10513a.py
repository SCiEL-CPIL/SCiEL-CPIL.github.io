import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 145
    
    # Try ranks from 9 upward
    for r in range(9, 19):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 200 if r <= 10 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            seed = attempt * 7919 + r * 104729 + 12345
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            def make_residuals(r):
                def residuals(params):
                    U_f = params[:n*m*r].reshape(n*m, r)
                    V_f = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_f = params[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    res = np.empty(n*m*p)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res[idx] = np.dot(U_f[i*m+j, :], V_f[j*p+k, :] * W_f[k*n+i, :]) - 1
                                idx += 1
                    return res
                return residuals
            
            res_func = make_residuals(r)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(res_func, x0, method='lm', max_nfev=3000)
            except Exception:
                continue
            
            if result.cost < 1e-14:
                U_sol = result.x[:n*m*r].reshape(n*m, r)
                V_sol = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_sol = result.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_sol[i*m+j, :], V_sol[j*p+k, :] * W_sol[k*n+i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-10:
                    found = True
                    # Check for half-integer entries
                    all_half_int = True
                    for mat in [U_sol, V_sol, W_sol]:
                        for row in mat:
                            for val in row:
                                if abs(val * 2 - round(val * 2)) > 1e-4:
                                    all_half_int = False
                                    break
                            if not all_half_int:
                                break
                        if not all_half_int:
                            break
                    
                    if all_half_int:
                        U_list = [[round(val*2)/2 for val in row] for row in U_sol]
                        V_list = [[round(val*2)/2 for val in row] for row in V_sol]
                        W_list = [[round(val*2)/2 for val in row] for row in W_sol]
                    else:
                        U_list = [[float(val) for val in row] for row in U_sol]
                        V_list = [[float(val) for val in row] for row in V_sol]
                        W_list = [[float(val) for val in row] for row in W_sol]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
    
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
