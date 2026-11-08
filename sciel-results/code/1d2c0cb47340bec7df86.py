import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 22
    target_rank = 22
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    # Else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def try_find_decomposition(rank, seed):
        np.random.seed(seed)
        # Initialize U, V, W
        # Scale appropriately
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Simple gradient descent or L-BFGS-B
        # Objective: sum (T - sum U_c V_c W_c)^2
        # Flatten T
        T_flat = T.flatten()
        
        def objective(params):
            U = params[:n*m*rank].reshape(n*m, rank)
            V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
            
            # Compute tensor product sum
            # This is expensive to do in a loop for every objective call if rank is large
            # But rank is 22, sizes are small.
            # Reshape U to (n*m, rank, 1), V to (m*p, rank, 1) ? No.
            # We want sum_c U[:,c] * V[:,c] * W[:,c]
            # This is an outer product sum.
            
            # Efficient way:
            # Tensor C = np.einsum('ic,jc,kc->ijk', U, V, W)
            C = np.einsum('ic,jc,kc->ijk', U, V, W)
            C_flat = C.flatten()
            diff = T_flat - C_flat
            return np.dot(diff, diff)
        
        # We need gradients for L-BFGS-B
        # Or we can use a simpler method if scipy is slow.
        # Let's use scipy.optimize.minimize with L-BFGS-B
        
        initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = scipy.optimize.minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
            if result.fun < 1e-8:
                # Extract U, V, W
                U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                V_sol = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                # Check if we can round to integers/half-integers
                # The problem allows integer/half-integer entries.
                # Let's see if the solution is close to such values.
                # For now, return the float solution if valid.
                # But the task says "integer/half-integer entries".
                # If the solution is not close to these, it might be invalid for the evaluator if it checks types strictly?
                # "integer/half-integer entries that exactly decompose"
                # So we must find a solution with such entries.
                
                # Rounding strategy:
                # Try rounding to nearest 0.5
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify
                C_int = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                if np.allclose(C_int, T, atol=1e-9):
                    return U_int, V_int, W_int
                else:
                    # Maybe it's integers?
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    C_int = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                    if np.allclose(C_int, T, atol=1e-9):
                        return U_int, V_int, W_int
                    else:
                        return None
            else:
                return None
        except Exception:
            return None

    import scipy.optimize
    
    start_time = time.time()
    best_result = None
    
    # Try a few seeds
    for seed in range(20):
        if time.time() - start_time > 120:
            break
        res = try_find_decomposition(target_rank, seed)
        if res is not None:
            U, V, W = res
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
            
    if best_result is None:
        # Fallback to schoolbook rank 24
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
        best_result = {"U": U, "V": V, "W": W}
        
    return best_result
