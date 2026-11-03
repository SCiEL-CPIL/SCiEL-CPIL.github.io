import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    n, m, p = 3, 3, 5
    
    # Target tensor T[a,b,c] where a in 0..8, b in 0..14, c in 0..14
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    best_result = None
    best_rank = n*m*p
    
    # Try ranks from 44 down to 35
    for r in range(44, 34, -1):
        if time.time() > deadline - 20.0:
            break
            
        # Try a few random restarts for this rank
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
                
            np.random.seed(seed + r)
            # Initialize U, V, W
            # Scale initialization to avoid overflow
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Minimize error
            # We can use a simple gradient descent or scipy
            # Given time, maybe a simple iterative approach or scipy minimize
            # Let's use scipy.optimize.minimize
            
            def objective(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Compute tensor T_est
                # T_est[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # This is an outer product sum
                # Efficient computation:
                # T_est = np.einsum('at,bt,ct->abc', U, V, W)
                T_est = np.einsum('at,bt,ct->abc', U, V, W)
                return np.sum((T_est - T)**2)
                
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9})
                
                if res.success or res.fun < 1e-6:
                    # Check if we can round to half-integers
                    x_opt = res.x
                    U_opt = x_opt[:n*m*r].reshape(n*m, r)
                    V_opt = x_opt[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_opt = x_opt[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    T_check = np.einsum('at,bt,ct->abc', U_round, V_round, W_round)
                    if np.allclose(T_check, T, atol=1e-9):
                        # Valid!
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
            except Exception as e:
                pass
                
    # Fallback to schoolbook
    rank = n*m*p
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
