import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 110
    
    # Build the target tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    schoolbook_rank = n * m * p  # 45
    
    # Schoolbook decomposition as fallback
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sch.tolist(), V_sch.tolist(), W_sch.tolist())
    best_rank = schoolbook_rank

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        approx = np.zeros_like(T)
        for r in range(rank):
            approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T, approx, atol=tol)

    def greedy_init(T, target_rank):
        """
        Greedy initialization:
        1. Copy T
        2. For each rank term, find the best rank-1 approximation to the residual
        3. Subtract the approximation
        """
        residual = T.copy()
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for r in range(target_rank):
            # Find the entry with the maximum absolute value in residual
            idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
            # This is a very crude greedy step. 
            # Better: Use SVD on a matrix unfolding to find the best rank-1 approximation.
            # Unfold residual into a matrix. 
            # Let's unfold along mode 1: (n*m) x (m*p * p*n)
            # This is huge. 
            # Alternative: Just pick the max element and set U, V, W to have 1 there, 0 elsewhere? 
            # That's just schoolbook.
            
            # Let's try a slightly smarter greedy:
            # Pick a random (a,b) where residual is non-zero?
            # Or, use the fact that T is sparse.
            
            # For now, let's stick to a random initialization mixed with schoolbook structure
            pass
        
        return U, V, W

    def try_rank(target_rank, max_attempts=5):
        """Try to find a decomposition of given rank using optimization."""
        if time.time() > deadline - 20:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialization
            if attempt == 0:
                # Use schoolbook columns
                U_init = U_sch[:, :target_rank].copy()
                V_init = V_sch[:, :target_rank].copy()
                W_init = W_sch[:, :target_rank].copy()
            elif attempt == 1:
                # Random
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            else:
                # Random with larger variance
                U_init = np.random.randn(n*m, target_rank)
                V_init = np.random.randn(m*p, target_rank)
                W_init = np.random.randn(p*n, target_rank)
            
            # Flatten parameters
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                approx = np.zeros_like(T)
                for r in range(target_rank):
                    approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
                
                return np.sum((T - approx)**2)
            
            try:
                from scipy.optimize import minimize
                # Use L-BFGS-B
                result = minimize(objective, params, method='L-BFGS-B',
                                 options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-10})
                
                if result.fun < 1e-6:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Try rounding to integers
                    U_int = np.round(U_r)
                    V_int = np.round(V_r)
                    W_int = np.round(W_r)
                    if verify_decomposition(U_int, V_int, W_int):
                        return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
                    # Try rounding to half-integers
                    U_half = np.round(U_r * 2) / 2
                    V_half = np.round(V_r * 2) / 2
                    W_half = np.round(W_r * 2) / 2
                    if verify_decomposition(U_half, V_half, W_half):
                        return (U_half.tolist(), V_half.tolist(), W_half.tolist())
                        
            except Exception:
                pass
        
        return None

    # Search for lower rank
    # We know rank >= 15.
    # Try ranks from 20 to 44.
    for target_rank in range(20, 45):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(target_rank, max_attempts=3)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # If we found one, we can stop or continue to find lower
            # Given time constraints, finding one is a big win.
            # Let's break if we find a rank significantly lower than 45
            if target_rank < 40:
                break

    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
