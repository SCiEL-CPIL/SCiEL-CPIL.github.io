import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 110.0
    
    n, m, p = 3, 3, 5
    size_U = n * m  # 9
    size_V = m * p  # 15
    size_W = p * n  # 15
    
    # Trivial rank-45 decomposition
    R0 = n * m * p  # 45
    U0 = np.zeros((size_U, R0))
    V0 = np.zeros((size_V, R0))
    W0 = np.zeros((size_W, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = R0
    
    # Try to find lower rank decompositions
    for R in [R0 - 1, R0 - 2]:  # Try rank 44, 43
        if time.time() - start_time > timeout - 10:
            break
        
        np.random.seed(42)
        U = U0[:, :R].copy() + 0.01 * np.random.randn(size_U, R)
        V = V0[:, :R].copy() + 0.01 * np.random.randn(size_V, R)
        W = W0[:, :R].copy() + 0.01 * np.random.randn(size_W, R)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x, R=R):
            if time.time() - start_time > timeout - 5:
                return 1e10
            Ux = x[:size_U*R].reshape(size_U, R)
            Vx = x[size_U*R:2*size_U*R].reshape(size_V, R)
            Wx = x[2*size_U*R:].reshape(size_W, R)
            
            # Vectorized computation
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(Ux[i*m+j, :], Vx[j*p+k, :] * Wx[k*n+i, :])
                        residual += (1.0 - val) ** 2
            return residual
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 3000})
            if result.fun < 1e-6:
                Ux = result.x[:size_U*R].reshape(size_U, R)
                Vx = result.x[size_U*R:2*size_U*R].reshape(size_V, R)
                Wx = result.x[2*size_U*R:].reshape(size_W, R)
                
                # Try rounding to half-integers
                Ux_r = np.round(Ux * 2) / 2
                Vx_r = np.round(Vx * 2) / 2
                Wx_r = np.round(Wx * 2) / 2
                
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(Ux_r[i*m+j, :], Vx_r[j*p+k, :] * Wx_r[k*n+i, :])
                            if abs(val - 1.0) > 1e-6:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    best_result = {"U": Ux_r.tolist(), "V": Vx_r.tolist(), "W": Wx_r.tolist()}
                    best_rank = R
                    break
        except:
            pass
    
    return best_result
