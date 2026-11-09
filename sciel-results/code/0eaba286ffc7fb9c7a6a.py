"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition with integer/half-integer entries."""
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    # Check tensor equation
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half(x):
    return np.round(x * 2) / 2

def _objective_and_grad(params, n, m, p, r):
    """Compute objective function and gradient for flattened parameters."""
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = params[:size_U].reshape(n*m, r)
    V = params[size_U:size_U+size_V].reshape(m*p, r)
    W = params[size_U+size_V:].reshape(p*n, r)
    
    error = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Precompute for efficiency
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for idx in range(r):
                    s += U[i*m+j, idx] * V[j*p+k, idx] * W[k*n+i, idx]
                err = s - 1.0
                error += err * err
                for idx in range(r):
                    grad_U[i*m+j, idx] += 2 * err * V[j*p+k, idx] * W[k*n+i, idx]
                    grad_V[j*p+k, idx] += 2 * err * U[i*m+j, idx] * W[k*n+i, idx]
                    grad_W[k*n+i, idx] += 2 * err * U[i*m+j, idx] * V[j*p+k, idx]
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return error, grad

def _try_rank(r, n, m, p, num_trials=5, time_limit=10):
    """Try to find a rank-r decomposition via random init + L-BFGS-B."""
    start = time.time()
    U_shape = (n*m, r)
    V_shape = (m*p, r)
    W_shape = (p*n, r)
    
    total_size = n*m*r + m*p*r + p*n*r
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        np.random.seed(trial * 12345 + 678)
        U = np.random.randn(*U_shape) * 0.5
        V = np.random.randn(*V_shape) * 0.5
        W = np.random.randn(*W_shape) * 0.5
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def fun(x):
            err, grad = _objective_and_grad(x, n, m, p, r)
            return err, grad
        
        try:
            res = scipy.optimize.minimize(fun, x0, method='L-BFGS-B', jac=True, 
                                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-8:
                # Extract optimized parameters
                x_opt = res.x
                U_opt = x_opt[:n*m*r].reshape(n*m, r)
                V_opt = x_opt[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_opt = x_opt[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Snap to half-integers
                U_snapped = _snap_to_half(U_opt)
                V_snapped = _snap_to_half(V_opt)
                W_snapped = _snap_to_half(W_opt)
                
                # Verify
                if _verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    
    # Try ranks from 12 upward (known lower bound is higher than 9 for 2x3x3)
    for r in range(12, 18):
        if time.time() - global_start > 130:
            break
        result = _try_rank(r, n, m, p, num_trials=5, time_limit=15)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
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
# EVOLVE-BLOCK-END
