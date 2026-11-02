"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Build the full tensor for verification
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return T
    
    T = build_tensor(n, m, p)
    
    def verify_decomp(U, V, W, n, m, p, T, tol=1e-9):
        """Check if decomposition is exact."""
        nm, m_p, p_n = n*m, m*p, p*n
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Compute predicted tensor
        pred = np.zeros((nm, m_p, p_n))
        for r_idx in range(r):
            pred += np.einsum('a,b,c->abc', U_mat[:, r_idx], V_mat[:, r_idx], W_mat[:, r_idx])
        
        return np.allclose(pred, T, atol=tol)

    # Try to find lower rank decomposition
    # We use a very short budget to avoid wasting time if the rank is minimal.
    np.random.seed(42)
    
    # Try ranks 59, 58, 57
    for target_rank in [59, 58, 57]:
        if time.time() > deadline - 15:
            break
        
        # Attempt to find a decomposition of target_rank
        # Use multiple random restarts
        max_restarts = 5 if target_rank == 59 else 3
        found = False
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*target_rank].reshape(n*m, target_rank)
                V = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                pred = np.zeros((n*m, m*p, p*n))
                for r_idx in range(target_rank):
                    pred += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                
                error = pred - T
                return np.sum(error**2)
            
            # Use L-BFGS-B which is good for large sparse problems, but here dense.
            # We can also just use BFGS if the dimension is small enough.
            # Dim = (12 + 20 + 15) * 59 = 47 * 59 = 2773. BFGS might be slow.
            # L-BFGS-B is better.
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-8:
                U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Verify
                if verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, T):
                    best_rank = target_rank
                    best_U = U_sol.tolist()
                    best_V = V_sol.tolist()
                    best_W = W_sol.tolist()
                    found = True
                    break
        
        if found:
            break

    # Final verification of best solution
    if best_rank < rank_schoolbook:
        if verify_decomp(best_U, best_V, best_W, n, m, p, T):
            return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return schoolbook as fallback
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
