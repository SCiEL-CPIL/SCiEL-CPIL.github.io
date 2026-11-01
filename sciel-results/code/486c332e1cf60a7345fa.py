import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = 2, 3, 4
    
    # Pre-compute target tensor
    # Shape (n, m, m, p)
    # T[i, j, l, k] = 1 if j == l else 0
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for l in range(m):
                for k in range(p):
                    if j == l:
                        T[i, j, l, k] = 1.0

    def residuals(params, rank):
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        U_3d = U_flat.reshape(n, m, rank)
        V_3d = V_flat.reshape(m, p, rank)
        W_3d = W_flat.reshape(p, n, rank)
        
        # S[i, j, l, k] = sum_t U[i, j, t] * V[l, k, t] * W[k, i, t]
        # U: i, j, t
        # V: l, k, t
        # W: k, i, t
        S = np.einsum('ijt,lkt,ki->ijlk', U_3d, V_3d, W_3d)
        
        res = S - T
        return res.flatten()

    def check(U, V, W, rank):
        # U, V, W are lists or arrays of shape (..., rank)
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        S = np.einsum('ijt,lkt,ki->ijlk', U_3d, V_3d, W_3d)
        
        # Check exact match
        if not np.allclose(S, T, atol=1e-9):
            return False
        return True

    # Try ranks from low to high
    for rank in [17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
            
        n_params = rank * (n*m + m*p + p*n)
        
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
                
            np.random.seed(attempt * 100 + rank)
            x0 = np.random.randn(n_params) * 0.5
            
            try:
                res = minimize(residuals, x0, args=(rank,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e-10:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    # Try rounding to integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if check(U_int, V_int, W_int, rank):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
                    # Try rounding to half-integers
                    U_half = np.round(U_opt * 2) / 2
                    V_half = np.round(V_opt * 2) / 2
                    W_half = np.round(W_opt * 2) / 2
                    if check(U_half, V_half, W_half, rank):
                        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                        
            except Exception:
                pass

    # Fallback to schoolbook
    rank = 24
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
