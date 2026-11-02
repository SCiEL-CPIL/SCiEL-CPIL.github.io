import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Try ranks from low to high
    for rank in range(1, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Use scipy to solve
            try:
                from scipy.optimize import least_squares
                from scipy.optimize import minimize
                
                def residuals(params):
                    U_r = params[:nm*rank].reshape(nm, rank)
                    V_r = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_r = params[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Compute the tensor product
                    # For each r, we have outer product U_r[:,r] ⊗ V_r[:,r] ⊗ W_r[:,r]
                    # Sum over r
                    # This is expensive, so let's compute the error differently
                    
                    # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # We can compute this as: for each r, add U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                    
                    # More efficient: T_approx = sum_r np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    # But that's still O(nm*mp*pn*rank)
                    
                    # Let's compute the error on all entries
                    error = np.zeros((nm, mp, pn))
                    for r in range(rank):
                        error += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
                    
                    diff = error - T
                    return diff.ravel()
                
                x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
                
                # This might be too slow for large rank. Let's limit.
                if rank > 20:
                    continue  # Skip large ranks for now, too slow
                
                result = least_squares(residuals, x0, method='trf', max_nfev=1000)
                
                if result.success or np.linalg.norm(result.fun) < 1e-6:
                    U_r = result.x[:nm*rank].reshape(nm, rank)
                    V_r = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_r = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Verify
                    error = np.zeros((nm, mp, pn))
                    for r in range(rank):
                        error += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
                    
                    if np.max(np.abs(error - T)) < 1e-6:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r)
                        break  # Found a valid decomposition at this rank
                
            except Exception:
                pass
        
        if best_rank < n * m * p and best_result is not None:
            break
    
    # If no better found, return schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
