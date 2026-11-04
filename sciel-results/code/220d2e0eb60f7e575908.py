import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 110
    
    # Build the system of equations
    # For each (i, j, l, k): sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta(j,l)
    
    def residuals(params, R):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
        W = params[n*m*R+m*p*R:].reshape(p*n, R)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        val = np.dot(U[i*m+j, :], V[l*p+k, :] * W[k*n+i, :])
                        expected = 1.0 if j == l else 0.0
                        res.append(val - expected)
        return np.array(res)
    
    # Try rank 17
    for R in [17, 16, 15]:
        if time.time() > deadline:
            break
        
        best_result = None
        for trial in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial)
            x0 = np.random.randn((n*m + m*p + p*n) * R) * 0.1
            
            try:
                result = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=1000)
                if result.success or result.cost < 1e-10:
                    U = result.x[0:n*m*R].reshape(n*m, R)
                    V = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    max_err = 0
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                for l in range(m):
                                    val = np.dot(U_r[i*m+j, :], V_r[l*p+k, :] * W_r[k*n+i, :])
                                    expected = 1.0 if j == l else 0.0
                                    max_err = max(max_err, abs(val - expected))
                    
                    if max_err < 1e-9:
                        best_result = (R, U_r, V_r, W_r)
                        break
            except Exception:
                pass
        
        if best_result is not None:
            R, U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook rank 18
    R = 18
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    for c, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
        U[i*m+j, c] = 1.0
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
