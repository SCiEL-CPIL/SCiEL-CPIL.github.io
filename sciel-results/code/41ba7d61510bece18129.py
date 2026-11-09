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
        
    def compute_T(U, V, W, R):
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return T

    def objective(x, R):
        if time.time() - start_time > timeout - 5:
            return 1e10
        Ux = x[:size_U*R].reshape(size_U, R)
        Vx = x[size_U*R:2*size_U*R].reshape(size_V, R)
        Wx = x[2*size_U*R:].reshape(size_W, R)
        T = compute_T(Ux, Vx, Wx, R)
        residual = np.sum((T - 1.0) ** 2)
        return residual

    for R in range(R0 - 1, 20, -1):
        if time.time() - start_time > timeout - 10:
            break
        
        found = False
        for attempt in range(10):
            if time.time() - start_time > timeout - 5:
                break
                
            np.random.seed(42 + attempt)
            U = np.random.randn(size_U, R) * 0.1
            V = np.random.randn(size_V, R) * 0.1
            W = np.random.randn(size_W, R) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 1000})
                
                if result.fun < 1e-6:
                    Ux = result.x[:size_U*R].reshape(size_U, R)
                    Vx = result.x[size_U*R:2*size_U*R].reshape(size_V, R)
                    Wx = result.x[2*size_U*R:].reshape(size_W, R)
                    
                    Ux_r = np.round(Ux * 2) / 2
                    Vx_r = np.round(Vx * 2) / 2
                    Wx_r = np.round(Wx * 2) / 2
                    
                    T_r = compute_T(Ux_r, Vx_r, Wx_r, R)
                    if np.allclose(T_r, 1.0, atol=1e-6):
                        best_result = {"U": Ux_r.tolist(), "V": Vx_r.tolist(), "W": Wx_r.tolist()}
                        best_rank = R
                        found = True
                        break
            except:
                pass
        
        if found:
            break
            
    return best_result
