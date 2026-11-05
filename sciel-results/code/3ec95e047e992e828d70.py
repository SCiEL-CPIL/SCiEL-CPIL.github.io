import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Target tensor values
    # T[i,j,k] = 1 if j==k else 0 (independent of l)
    
    # We'll search for rank r decomposition
    # Parameters: U (n*m x r), V (m*p x r), W (p*n x r)
    # Total params: (n*m + m*p + p*n) * r = (9 + 12 + 12) * r = 33r
    
    # Residual: for each (i,j,k), residual = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - delta_{j,k}
    # Number of residuals: n*m*p = 3*3*4 = 36 (but only 27 are independent since T is independent of l)
    # Actually, T[i,j,k,l] = delta_{j,k} for all l, so we have 3*3*3*4 = 108 residuals, but they're redundant.
    # Let's just use the 27 unique (i,j,k) constraints.
    
    def residual(params, r):
        U = params[0:n*m*r].reshape(n*m, r)
        V = params[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
        W = params[(n*m+m*p)*r:].reshape(p*n, r)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    target = 1.0 if j == k else 0.0
                    res.append(val - target)
        return np.array(res)
    
    # Try ranks from 3 to 36
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for r in range(3, 37):
        if time.time() - start_time > 110:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() - start_time > 110:
                break
            
            np.random.seed(attempt * 1000 + r)
            params0 = np.random.randn((n*m + m*p + p*n) * r)
            
            try:
                result = least_squares(lambda x: residual(x, r), params0, method='lm', max_nfev=10000)
                if result.cost < 1e-10:
                    U = result.x[0:n*m*r].reshape(n*m, r)
                    V = result.x[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
                    W = result.x[(n*m+m*p)*r:].reshape(p*n, r)
                    
                    # Check if entries are close to half-integers
                    all_half = all(abs(2*x - round(2*x)) < 1e-6 for x in np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
                    if all_half:
                        # Convert to half-integers
                        U = np.round(2*U)/2
                        V = np.round(2*V)/2
                        W = np.round(2*W)/2
                        best_rank = r
                        best_U = U
                        best_V = V
                        best_W = W
                        break
            except:
                pass
        
        if best_rank == r:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
