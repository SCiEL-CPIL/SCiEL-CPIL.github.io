import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of size (n*m, m*p, p*n)
    # Indices: a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if j from a == j from b and k from b == k from c and i from a == i from c
    
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    T = np.zeros((size_a, size_b, size_c), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # We want to find U, V, W of rank R such that sum_r U[:,r] V[:,r] W[:,r] approx T
    # Let's try rank 49
    R = 49
    
    start_time = time.time()
    time_limit = 140.0 # Leave 10s margin
    
    best_U = None
    best_V = None
    best_W = None
    best_loss = float('inf')
    
    # Function to evaluate loss
    def loss(params):
        # params is a flattened array of U, V, W
        # U: size_a * R, V: size_b * R, W: size_c * R
        U_flat = params[:size_a * R].reshape(size_a, R)
        V_flat = params[size_a * R : size_a * R + size_b * R].reshape(size_b, R)
        W_flat = params[size_a * R + size_b * R :].reshape(size_c, R)
        
        # Compute tensor product sum
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Efficient computation:
        # T_approx = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        T_approx = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        
        return np.sum((T - T_approx)**2)
        
    # Gradient of loss
    def gradient(params):
        U_flat = params[:size_a * R].reshape(size_a, R)
        V_flat = params[size_a * R : size_a * R + size_b * R].reshape(size_b, R)
        W_flat = params[size_a * R + size_b * R :].reshape(size_c, R)
        
        T_approx = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        diff = T_approx - T
        
        # dL/dU[a, r] = sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
        grad_U = np.einsum('abc,br,cr->ar', diff, V_flat, W_flat)
        grad_V = np.einsum('abc,ar,cr->br', diff, U_flat, W_flat)
        grad_W = np.einsum('abc,ar,br->cr', diff, U_flat, V_flat)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
    # Try a few random initializations
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random values
        U_init = np.random.randn(size_a, R) * 0.1
        V_init = np.random.randn(size_b, R) * 0.1
        W_init = np.random.randn(size_c, R) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(loss, params_init, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6})
            
            if res.success or res.fun < best_loss:
                best_loss = res.fun
                if res.fun < 1e-6: # Found a very good solution
                    U_best = res.x[:size_a * R].reshape(size_a, R)
                    V_best = res.x[size_a * R : size_a * R + size_b * R].reshape(size_b, R)
                    W_best = res.x[size_a * R + size_b * R :].reshape(size_c, R)
                    
                    # Check if entries are close to integers or half-integers
                    # The Winograd decomposition has entries in {0, 1, -1, 0.5, -0.5}
                    # Let's try to snap to these values
                    U_snapped = np.round(U_best * 2) / 2
                    V_snapped = np.round(V_best * 2) / 2
                    W_snapped = np.round(W_best * 2) / 2
                    
                    # Verify snapped solution
                    T_snapped = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
                    err_snapped = np.sum((T - T_snapped)**2)
                    
                    if err_snapped < 1e-6:
                        # Valid integer/half-integer decomposition found
                        return {
                            "U": U_snapped.tolist(),
                            "V": V_snapped.tolist(),
                            "W": W_snapped.tolist()
                        }
                        
        except Exception as e:
            continue

    # If no rank 49 found, return schoolbook rank 64
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
