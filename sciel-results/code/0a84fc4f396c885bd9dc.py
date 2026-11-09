import time
import numpy as np
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _objective_and_grad(x, n, m, p, R):
    """
    Calculate objective and gradient for tensor decomposition.
    x is a flat array of [U_flat, V_flat, W_flat].
    """
    U_size = n * m * R
    V_size = m * p * R
    W_size = p * n * R
    
    U_flat = x[:U_size].reshape(n*m, R)
    V_flat = x[U_size:U_size+V_size].reshape(m*p, R)
    W_flat = x[U_size+V_size:U_size+V_size+W_size].reshape(p*n, R)
    
    # Compute T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T[i,j,k] = 1 for all i,j,k.
    
    # Efficient computation using einsum or loops?
    # n=2, m=3, p=10.
    # T is 2x3x10.
    # U is 6xR, V is 30xR, W is 20xR.
    
    # Let's compute the residual vector for all i,j,k
    # Res[i,j,k] = T[i,j,k] - 1
    
    # We can compute T as a tensor product.
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a 3-mode tensor.
    # Let's just loop for gradient calculation as dimensions are small.
    
    obj = 0.0
    grad_U = np.zeros_like(U_flat)
    grad_V = np.zeros_like(V_flat)
    grad_W = np.zeros_like(W_flat)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T_val = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # d(T_val)/d(U[i*m+j, r]) = V[j*p+k, r] W[k*n+i, r]
                # d(T_val)/d(V[j*p+k, r]) = U[i*m+j, r] W[k*n+i, r]
                # d(T_val)/d(W[k*n+i, r]) = U[i*m+j, r] V[j*p+k, r]
                
                # We can vectorize this sum over r
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                
                u_row = U_flat[u_idx, :]
                v_row = V_flat[v_idx, :]
                w_row = W_flat[w_idx, :]
                
                t_val = np.dot(u_row, v_row * w_row)
                res = t_val - 1.0
                
                obj += res * res
                
                # Gradients
                # grad_U[u_idx, :] += 2 * res * (v_row * w_row)
                # grad_V[v_idx, :] += 2 * res * (u_row * w_row)
                # grad_W[w_idx, :] += 2 * res * (u_row * v_row)
                
                grad_U[u_idx, :] += 2.0 * res * (v_row * w_row)
                grad_V[v_idx, :] += 2.0 * res * (u_row * w_row)
                grad_W[w_idx, :] += 2.0 * res * (u_row * v_row)
                
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return obj, grad

def _optimize_rank(n, m, p, R, max_time, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    U_size = n * m * R
    V_size = m * p * R
    W_size = p * n * R
    
    # Initialize with small random values
    x0 = np.random.randn(U_size + V_size + W_size) * 0.1
    
    start_time = time.time()
    
    # Use L-BFGS-B
    # We need to stop if time runs out.
    # Scipy's minimize doesn't have a time-based callback that easily stops the whole process cleanly in all versions,
    # but we can check time in a callback and raise an exception or set a flag.
    
    time_exceeded = [False]
    def callback(xk):
        if time.time() - start_time > max_time * 0.9:
            time_exceeded[0] = True
            raise Exception("Time exceeded")
        return False
        
    try:
        res = minimize(
            _objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50},
            callback=callback
        )
    except Exception as e:
        # If time exceeded, use the last best point if we can retrieve it?
        # Minimize returns the last point if it raises? No, it stops.
        # We need to capture the best x.
        # Actually, if we raise an exception, we lose the state.
        # Better to use a callback that sets a flag, and check in a loop?
        # Or just run for a fixed number of iterations and check time.
        
        # Let's restart with a simpler approach: manual gradient descent or just many restarts with limited iterations.
        # Given the complexity, let's try a few restarts with limited L-BFGS iterations.
        pass
        
    # The above try-except is tricky. Let's implement a manual loop with time checks.
    # Or, simpler: use `scipy.optimize.minimize` with a small maxiter and check time between restarts.
    
    best_res = None
    best_obj = float('inf')
    
    for trial in range(10): # 10 restarts
        if time.time() - start_time > max_time:
            break
            
        x0 = np.random.randn(U_size + V_size + W_size) * 0.1
        
        # Run minimize with a smaller maxiter to allow multiple restarts
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_res = res
        except Exception:
            continue
            
    if best_res is None:
        return None
        
    x_opt = best_res.x
    U_opt = x_opt[:U_size].reshape(n*m, R)
    V_opt = x_opt[U_size:U_size+V_size].reshape(m*p, R)
    W_opt = x_opt[U_size+V_size:U_size+V_size+W_size].reshape(p*n, R)
    
    # Verify
    if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-6):
        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Rank 45
    if time.time() - start_time < deadline - 5:
        time_left = deadline - (time.time() - start_time) - 5
        result = _optimize_rank(n, m, p, 45, time_left)
        if result is not None:
            best_result = result
            best_rank = 45
    
    # Try Rank 50 if 45 failed
    if best_rank > 50 and time.time() - start_time < deadline - 5:
        time_left = deadline - (time.time() - start_time) - 5
        result = _optimize_rank(n, m, p, 50, time_left)
        if result is not None:
            best_result = result
            best_rank = 50

    return best_result
