import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build the target tensor T[a,b,c]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Build Schoolbook Decomposition (Rank 45)
    schoolbook_rank = n * m * p
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

    # Try to find a lower rank decomposition
    # Search from 41 down to 35
    # Use optimizer with better initialization
    
    def verify_decomposition(U, V, W, atol=1e-10):
        # Check nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > atol:
                        return False
        # Check zero entries (sample)
        np.random.seed(42)
        for _ in range(100):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            # Skip nonzero entries
            i = a // m
            j = a % m
            k = b % p
            if i == a // m and j == a % m and k == b % p and c == k*n + i:
                continue
            val = 0.0
            for r in range(U.shape[1]):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val) > atol:
                return False
        return True

    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 10:
            break
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(100 + restart)
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            c = k*n+i
                            val = 0.0
                            for r in range(target_rank):
                                val += U_r[a, r] * V_r[b, r] * W_r[c, r]
                            residual += (val - 1.0) ** 2
                return residual

            try:
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                 options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-12})
                
                if result.fun < 1e-10:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        best_result = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                        best_rank = target_rank
                        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
            except Exception:
                pass

    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
