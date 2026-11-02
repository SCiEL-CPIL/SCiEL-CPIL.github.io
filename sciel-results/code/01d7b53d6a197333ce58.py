import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    target_rank = max(n * m, m * p, p * n)
    
    # Build the target tensor T
    # T[i, j, k] = 1 for all i, j, k
    # The decomposition is sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    # We vectorize the indices.
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    def flatten_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We need to compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is not a simple matrix multiplication.
        # We can iterate over r? No, that's slow.
        # We can use einsum.
        # U[i, j, r] V[j, k, r] -> M[i, k, r] ? No, j is summed.
        # temp[i, k, r] = sum_j U[i, j, r] V[j, k, r]
        # Then result[i, k] = sum_r temp[i, k, r] W[k, i, r]
        # Note: W index is k, i.
        
        # Reshape U to (n, m, R)
        # Reshape V to (m, p, R)
        # Reshape W to (p, n, R)
        
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Compute sum_j U[i, j, r] V[j, k, r]
        # einsum('ijr,jkr->ikr', U_3d, V_3d)
        temp = np.einsum('ijr,jkr->ikr', U_3d, V_3d)
        
        # Compute sum_r temp[i, k, r] W[k, i, r]
        # W_3d is (p, n, R). We need W[k, i, r].
        # So we need to transpose W_3d to (n, p, R) or access correctly.
        # W_3d[k, i, r].
        # temp is (n, p, R).
        # result[i, k] = sum_r temp[i, k, r] W_3d[k, i, r]
        # We can swap axes of W_3d to (n, p, R) -> W_3d.T (axis 0,1)
        W_trans = W_3d.transpose(1, 0, 2) # (n, p, R)
        
        result = np.einsum('ikr,ikr->ik', temp, W_trans)
        
        # result is (n, p). We need to check against 1s.
        return result

    def residuals(params):
        R = len(params) // (n * m + m * p + p * n)
        U = params[:n*m].reshape(n, m, R)
        V = params[n*m:n*m+m*p].reshape(m, p, R)
        W = params[n*m+m*p:].reshape(p, n, R)
        
        # Compute T
        T = flatten_tensor(U.reshape(n*m, R), V.reshape(m*p, R), W.reshape(p*n, R))
        
        # Target is all ones
        target = np.ones((n, p))
        return (T - target).flatten()

    # Try to find a low rank solution
    best_solution = None
    best_error = float('inf')
    
    start_time = time.time()
    timeout = 100  # Leave 20s margin
    
    # Try rank 28
    R = target_rank
    num_vars = R * (n * m + m * p + p * n)
    
    # Random restarts
    max_restarts = 10
    for _ in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        x0 = np.random.randn(num_vars) * 0.1
        # Scale to help convergence
        x0 = x0 / np.linalg.norm(x0) * np.sqrt(num_vars)
        
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if res.fun is not None and np.max(np.abs(res.fun)) < 1e-5:
                # Check if we can round to integers
                # The solution might be close to integers
                x_sol = res.x
                # Extract U, V, W
                U_flat = x_sol[:n*m]
                V_flat = x_sol[n*m:n*m+m*p]
                W_flat = x_sol[n*m+m*p:]
                
                # Check max deviation from nearest integer/half-integer
                # For now, just check if it's a valid solution
                # We will try to round
                U_rounded = np.round(U_flat * 2) / 2
                V_rounded = np.round(V_flat * 2) / 2
                W_rounded = np.round(W_flat * 2) / 2
                
                U_r = U_rounded.reshape(n, m, R)
                V_r = V_rounded.reshape(m, p, R)
                W_r = W_rounded.reshape(p, n, R)
                
                T_r = flatten_tensor(U_r.reshape(n*m, R), V_r.reshape(m*p, R), W_r.reshape(p*n, R))
                err_r = np.max(np.abs(T_r - 1))
                
                if err_r < 1e-9:
                    return {
                        "U": U_r.reshape(n*m, R).tolist(),
                        "V": V_r.reshape(m*p, R).tolist(),
                        "W": W_r.reshape(p*n, R).tolist()
                    }
                    
                # If rounding failed, keep the float solution if error is very small
                # But the task requires exact.
                # We will store the best float solution as fallback if we can't find exact
                if np.max(np.abs(res.fun)) < best_error:
                    best_error = np.max(np.abs(res.fun))
                    best_solution = (U_flat, V_flat, W_flat)
                    
        except Exception as e:
            pass

    # If we found a good float solution, try to return it?
    # The task says "integer/half-integer entries".
    # If we don't have exact, we must return schoolbook.
    
    # Schoolbook fallback
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
