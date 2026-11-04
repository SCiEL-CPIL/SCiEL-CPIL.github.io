import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Build the target tensor T[a,b,c] where:
    # a = i*m + j (input A index)
    # b = j*p + k (input B index)  
    # c = k*n + i (output C index, trace convention)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    schoolbook_rank = n * m * p  # 45
    
    # Build schoolbook decomposition as fallback
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
    
    # Lower bound: max of the three mode dimensions
    lower_bound = max(n*m, m*p, p*n)  # max(9, 15, 15) = 15
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T using vectorization"""
        try:
            X = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(X, T, atol=atol)
        except Exception:
            return False

    def try_rank(target_rank, max_restarts=5):
        """Try to find a decomposition of given rank using vectorized objective."""
        nonlocal best_result, best_rank
        if time.time() > deadline - 15:
            return False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return False
            
            # Different initialization strategies
            if restart == 0:
                # Structured: use first target_rank columns of schoolbook
                U_init = U_sch[:, :target_rank].copy()
                V_init = V_sch[:, :target_rank].copy()
                W_init = W_sch[:, :target_rank].copy()
            elif restart == 1:
                # Random with small scale
                np.random.seed(42 + restart)
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            else:
                # Random with various scales
                np.random.seed(100 + restart * 7)
                scale = 0.3 + 0.1 * restart
                U_init = np.random.randn(n*m, target_rank) * scale
                V_init = np.random.randn(m*p, target_rank) * scale
                W_init = np.random.randn(p*n, target_rank) * scale
            
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            def objective(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Vectorized reconstruction
                X = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                residual = np.sum((X - T) ** 2)
                return residual
            
            try:
                # Use L-BFGS-B with vectorized objective
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                 options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-15, 'gtol': 1e-15})
                
                if result.fun < 1e-10:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        best_result = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                        best_rank = target_rank
                        return True
                    else:
                        # Check if unrounded is valid and close to half-integers
                        if verify_decomposition(U_r, V_r, W_r):
                            max_dev = max(
                                np.max(np.abs(U_r - np.round(U_r * 2) / 2)),
                                np.max(np.abs(V_r - np.round(V_r * 2) / 2)),
                                np.max(np.abs(W_r - np.round(W_r * 2) / 2))
                            )
                            if max_dev < 1e-6:
                                best_result = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                                best_rank = target_rank
                                return True
            except Exception:
                pass
        
        return False
    
    # Search from lower bound up to schoolbook
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        if try_rank(target_rank):
            # Found a decomposition at this rank. 
            # Since we are searching upwards, the first one found is the minimal in this search order.
            break
            
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
