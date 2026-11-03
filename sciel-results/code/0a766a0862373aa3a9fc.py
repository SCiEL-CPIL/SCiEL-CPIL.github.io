import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.flatten()

    def objective_and_grad(vars_flat, R):
        size_U = n*m*R
        size_V = m*p*R
        size_W = p*n*R
        
        U_flat = vars_flat[:size_U]
        V_flat = vars_flat[size_U:size_U+size_V]
        W_flat = vars_flat[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        S = np.einsum('ar,br,cr->abc', U, V, W)
        residual = S.flatten() - T_flat
        val = np.dot(residual, residual)
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return val, grad_flat

    def try_rank(R, max_restarts=50, time_limit=110):
        start_time = time.time()
        best_val = np.inf
        best_vars = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            # Use a mix of scales
            scale = 1.0
            if restart % 3 == 1: scale = 0.1
            if restart % 3 == 2: scale = 10.0
            
            np.random.seed(restart * 1000 + 42)
            U_init = np.random.randn(n*m, R) * scale
            V_init = np.random.randn(m*p, R) * scale
            W_init = np.random.randn(p*n, R) * scale
            
            vars_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective_and_grad, vars_init, args=(R,), method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_vars = res.x
            except Exception:
                continue
        
        if best_vars is None:
            return None
            
        size_U = n*m*R
        size_V = m*p*R
        size_W = p*n*R
        
        U = best_vars[:size_U].reshape(n*m, R)
        V = best_vars[size_U:size_U+size_V].reshape(m*p, R)
        W = best_vars[size_U+size_V:].reshape(p*n, R)
        
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        S = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        if np.allclose(S, T, atol=1e-9):
            return U_rounded, V_rounded, W_rounded
            
        return None

    # Try R=12
    result = try_rank(12)
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
