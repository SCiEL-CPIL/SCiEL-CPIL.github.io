import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank solutions
    # Lower bound is max(n*m, m*p, p*n) = 15
    min_rank = max(n*m, m*p, p*n)
    
    for r in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this rank
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W with random values
            # Use small values to help convergence?
            U_init = np.random.randn(n, m, r)
            V_init = np.random.randn(m, p, r)
            W_init = np.random.randn(p, n, r)
            
            # Flatten
            x = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*r].reshape(n, m, r)
                V = x[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
                W = x[n*m*r + m*p*r:].reshape(p, n, r)
                
                # Compute S_ijk = sum_c U_ijk V_jkc W_kic
                # Using einsum
                S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                return np.sum((S - 1.0)**2)
            
            # Use scipy optimize
            from scipy.optimize import minimize
            
            try:
                res = minimize(objective, x, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.success or res.fun < 1e-8:
                    x_opt = res.x
                    U_opt = x_opt[:n*m*r].reshape(n, m, r)
                    V_opt = x_opt[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
                    W_opt = x_opt[n*m*r + m*p*r:].reshape(p, n, r)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify
                    S_snap = np.einsum('ijc,jkc,kic->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(S_snap, 1.0, atol=1e-9):
                        # Convert to list of lists
                        U_list = U_snap.flatten().reshape(n*m, r).tolist()
                        V_list = V_snap.flatten().reshape(m*p, r).tolist()
                        W_list = W_snap.flatten().reshape(p*n, r).tolist()
                        
                        best_solution = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        found = True
                        break # Break attempt loop
            except Exception:
                pass
        
        # If we found a solution for this rank, we can stop?
        # No, we want the MINIMAL rank.
        # So we continue to lower ranks?
        # The loop is `for r in range(min_rank, best_rank)`.
        # If we found r=15, best_rank becomes 15. Loop ends.
        # If we found r=16, best_rank becomes 16. Loop continues to 17?
        # No, we want to find the MINIMAL rank.
        # So we should iterate r from min_rank upwards.
        # If we find a solution for r, that is the minimal rank (since we started from min).
        # So if we find a solution, we can return immediately?
        # Yes, because we are checking in increasing order of r.
        if found:
            return best_solution

    return best_solution

# EVOLVE-BLOCK-END
