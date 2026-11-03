import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def compute_T(U, V, W, R):
        """Compute T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]"""
        T = np.zeros((n, m, p))
        for r in range(R):
            Ur = U[:, r]  # (n*m,)
            Vr = V[:, r]  # (m*p,)
            Wr = W[:, r]  # (p*n,)
            for i in range(n):
                for j in range(m):
                    u_val = Ur[i * m + j]
                    for k in range(p):
                        T[i, j, k] += u_val * Vr[j * p + k] * Wr[k * n + i]
        return T
    
    def verify(U, V, W, tol=1e-8):
        R = U.shape[1]
        T = compute_T(U, V, W, R)
        return np.allclose(T, 1.0, atol=tol)
    
    def objective_and_grad(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = params[:size_U].reshape((n*m, R))
        V = params[size_U:size_U + size_V].reshape((m*p, R))
        W = params[size_U + size_V:].reshape((p*n, R))
        
        T = compute_T(U, V, W, R)
        error = np.sum((T - 1.0) ** 2)
        
        # Gradient
        grad = np.zeros_like(params)
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = 2.0 * (T[i, j, k] - 1.0)
                        u_val = U[i*m + j, r]
                        v_val = V[j*p + k, r]
                        w_val = W[k*n + i, r]
                        
                        grad[(i*m + j) * R + r] += diff * v_val * w_val
                        grad[size_U + (j*p + k) * R + r] += diff * u_val * w_val
                        grad[size_U + size_V + (k*n + i) * R + r] += diff * u_val * v_val
        
        return error, grad
    
    def search_rank(R, max_restarts=10):
        best_result = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart * 13)
            
            # Better initialization: use random but structured
            init_U = np.random.randn(n*m, R) * 0.5
            init_V = np.random.randn(m*p, R) * 0.5
            init_W = np.random.randn(p*n, R) * 0.5
            
            params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                result = minimize(
                    lambda x: objective_and_grad(x, R)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                
                if result.fun < best_error:
                    best_error = result.fun
                    if best_error < 1e-10:
                        U = result.x[:n*m*R].reshape((n*m, R))
                        V = result.x[n*m*R:n*m*R+m*p*R].reshape((m*p, R))
                        W = result.x[n*m*R+m*p*R:].reshape((p*n, R))
                        if verify(U, V, W, tol=1e-8):
                            return U, V, W
            except Exception:
                pass
        
        return None
    
    # Try ranks from 30 down to 23
    for R in range(30, 22, -1):
        if time.time() > deadline - 20:
            break
        
        result = search_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
