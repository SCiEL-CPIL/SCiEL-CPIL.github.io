import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower rank decompositions
    # Lower bound is roughly nm + mp + np - min(n,m,p) = 12+20+15-3 = 44
    for rank in range(59, 43, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. 1/sqrt(rank) is a common heuristic.
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            # Objective function
            def objective(params):
                u = params[:n*m*rank].reshape(n*m, rank)
                v = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                w = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                # Compute T_hat
                # T_hat[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
                # Efficient computation:
                # For each c, outer product of U[:,c], V[:,c], W[:,c]
                # Sum them up.
                # This is slow in pure Python loops, but numpy vectorization helps.
                # T_hat = np.einsum('ic,jc,kc->ijk', u, v, w)
                T_hat = np.einsum('ic,jc,kc->ijk', u, v, w)
                return np.sum((T - T_hat)**2)
                
            # Use scipy minimize
            try:
                from scipy.optimize import minimize
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
                
                if res.success or res.fun < 1e-6:
                    # Extract U, V, W
                    u = res.x[:n*m*rank].reshape(n*m, rank)
                    v = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    w = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    # Check if it's close to integers or half-integers
                    # Multiply by 2 to check if integers
                    U2 = u * 2
                    V2 = v * 2
                    W2 = w * 2
                    
                    # Check closeness to integers
                    def is_close_to_int(arr, tol=1e-4):
                        return np.all(np.abs(arr - np.round(arr)) < tol)
                    
                    if is_close_to_int(U2) and is_close_to_int(V2) and is_close_to_int(W2):
                        # Snap to integers / 2
                        U_int = np.round(U2) / 2.0
                        V_int = np.round(V2) / 2.0
                        W_int = np.round(W2) / 2.0
                        
                        # Verify exactness
                        T_hat = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                        if np.allclose(T_hat, T, atol=1e-9):
                            best_rank = rank
                            best_U = U_int
                            best_V = V_int
                            best_W = W_int
                            # Break early if we hit the theoretical lower bound?
                            # Or just keep going to see if we can go lower.
                            # Since we are iterating down, we found a better one.
                            # We can break the attempt loop for this rank?
                            # No, maybe we can find an even better one with same rank?
                            # Rank is the score. Lower is better.
                            # So we can break the attempt loop for this rank.
                            break
            except Exception:
                pass

    # Return the best found
    # Convert to lists of lists as required? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The interface expects {'U', 'V', 'W'}
    # The seed returns lists of lists.
    # I should return lists of lists to be safe, or numpy arrays?
    # "Keep the function names and signatures of the seed program."
    # Seed returns {"U": U, "V": V, "W": W} where U is list of lists.
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
