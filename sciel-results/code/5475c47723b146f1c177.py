import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor: T[i', j', j'', k', i, k] = 1 if i'=i, k'=k, j'=j'' else 0
    # We want U, V, W such that sum_r U[i'j', r] V[j''k', r] W[ik, r] = T[i'j'j''k'ik]
    
    # Indices
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i  (Note: task says W[k*n+i, r])
    
    # Precompute target values for all 729 combinations
    # i', j', j'', k', i, k
    targets = np.zeros((n, m, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets[i, j, j, k, i, k] = 1.0
                
    # Flatten targets to 1D array of length n*m*m*p*n*p
    # Order: i', j', j'', k', i, k
    target_flat = targets.flatten()
    
    def residual(x, R):
        # x is flattened U, V, W
        # U: n*m*R
        # V: m*p*R
        # W: p*n*R
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        # Compute tensor product
        # We need sum_r U[i'j', r] V[j''k', r] W[ik, r]
        # This is a batched matrix multiplication or einsum
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # We want result[i', j', j'', k', i, k]
        
        # Reshape U to (n, m, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R) # W index is k*n+i, so W[k, i, r]
        
        # Compute sum over r
        # term = U[i', j', r] * V[j'', k', r] * W[i, k, r]
        # This is tricky with einsum because indices are mixed.
        
        # Let's use np.einsum
        # U[i', j', r] -> 'ijr'
        # V[j'', k', r] -> 'jkr' (using j'' as j, k' as k)
        # W[i, k, r] -> 'ikr'
        
        # We want sum_r U[i'j'r] V[j''k'r] W[ikr]
        # Result indices: i', j', j'', k', i, k
        # einsum('ijr,jkr,ikr->ijjkik', U_reshaped, V_reshaped, W_reshaped)
        
        res = np.einsum('ijr,jkr,ikr->ijjkik', U_reshaped, V_reshaped, W_reshaped)
        return (res.flatten() - target_flat)

    def try_rank(R, max_restarts=5, timeout=10):
        start_time = time.time()
        for _ in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            # Random initialization
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R)
            
            # Optional: Scale to help convergence
            # x0 *= 0.1
            
            try:
                res = least_squares(residual, x0, args=(R,), method='lm', max_nfev=1000)
                if res.success and np.max(np.abs(res.fun)) < 1e-8:
                    U = res.x[:n*m*R].reshape(n*m, R)
                    V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    return U, V, W
            except Exception:
                continue
        return None

    # Try rank 20
    for R in [20, 21, 22, 23]:
        result = try_rank(R)
        if result:
            U, V, W = result
            # Convert to lists for return
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook rank 27
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
