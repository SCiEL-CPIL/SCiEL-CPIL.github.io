import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s limit
    
    # Target tensor for 2x3 * 3x9
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (27, rank)
    # W shape: (p*n, rank) = (18, rank)
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    target_indices = []
    target_vals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                target_vals.append(1.0)
    
    target_indices = np.array(target_indices)
    target_vals = np.array(target_vals)
    
    best_rank = n * m * p  # 54
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 30 to 53
    for r in range(30, 54):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_restarts = 20
        restarts_done = 0
        
        while restarts_done < max_restarts and not found:
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(restarts_done * 1000 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Scale to reasonable magnitude
            U /= np.sqrt(r)
            V /= np.sqrt(r)
            W /= np.sqrt(r)
            
            def objective(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Compute error only on target indices
                # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                err = 0.0
                for t in range(r):
                    u_col = U_p[:, t]
                    v_col = V_p[:, t]
                    w_col = W_p[:, t]
                    # For each target index (a,b,c), contribution is u_col[a]*v_col[b]*w_col[c]
                    # Vectorized:
                    contrib = u_col[target_indices[:, 0]] * v_col[target_indices[:, 1]] * w_col[target_indices[:, 2]]
                    err += np.sum(contrib * contrib)
                
                # We want sum of (sum_t U[a,t]*V[b,t]*W[c,t] - 1)^2
                # = sum_t sum_t' sum over indices of U[a,t]V[b,t]W[c,t] U[a,t']V[b,t']W[c,t'] - 2*sum_t sum over indices of U[a,t]V[b,t]W[c,t] + sum over indices of 1
                # This is getting complex. Let me use a simpler approach.
                
                # Actually, let's just compute the full reconstruction for target indices
                T_vals = np.zeros(len(target_indices))
                for t in range(r):
                    T_vals += U_p[target_indices[:, 0], t] * V_p[target_indices[:, 1], t] * W_p[target_indices[:, 2], t]
                
                diff = T_vals - 1.0
                return np.sum(diff * diff)
            
            p0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, p0, method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-5, 'maxfun': 300})
                
                if res.fun < 1e-6:
                    # Try to round to half-integers
                    U_r = np.round(res.x[:n*m*r].reshape(n*m, r) * 2) / 2
                    V_r = np.round(res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r) * 2) / 2
                    W_r = np.round(res.x[n*m*r + m*p*r:].reshape(p*n, r) * 2) / 2
                    
                    # Verify
                    err = 0.0
                    for (a, b, c) in target_indices:
                        val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                        err += (val - 1.0) ** 2
                    
                    if err < 1e-9:
                        best_rank = r
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                pass
            
            restarts_done += 1
        
        if found:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
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
