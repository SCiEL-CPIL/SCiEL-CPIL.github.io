import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def objective(params):
        r = len(params) // (n*m + m*p + p*n)
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
        W = params[(n*m+m*p)*r:].reshape(p*n, r)
        
        total = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+i, :])
                            expected = 1.0 if (j == jp and k == kp) else 0.0
                            total += (s - expected)**2
        return total
    
    def try_round(U, V, W):
        U_r = np.round(2*U) / 2
        V_r = np.round(2*V) / 2
        W_r = np.round(2*W) / 2
        # Check
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            s = np.sum(U_r[i*m+j, :] * V_r[jp*p+k, :] * W_r[kp*n+i, :])
                            expected = 1.0 if (j == jp and k == kp) else 0.0
                            if abs(s - expected) > 1e-9:
                                return None, None, None, False
        return U_r, V_r, W_r, True
    
    # Try gradient descent for various ranks
    best = None
    
    for r in [18, 21, 24, 27, 30, 36, 42]:
        if time.time() > deadline - 30:
            break
        
        for restart in range(2):
            if time.time() > deadline - 30:
                break
            
            size = (n*m + m*p + p*n) * r
            x0 = np.random.randn(size) * 0.1
            
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U = result.x[:n*m*r].reshape(n*m, r)
                V = result.x[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
                W = result.x[(n*m+m*p)*r:].reshape(p*n, r)
                
                U_r, V_r, W_r, ok = try_round(U, V, W)
                if ok:
                    best = (U_r, V_r, W_r, r)
                    break
        
        if best is not None:
            break
    
    if best is not None:
        U, V, W, r = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W, rank = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
