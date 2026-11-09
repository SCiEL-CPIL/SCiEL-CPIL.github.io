import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    random.seed(42)
    
    # Precompute index arrays for vectorized verification and updates
    # U update: for each (i,j), need V[j, :] and W[:, i]
    # V update: for each (j,k), need U[:, j] and W[k, :]
    # W update: for each (k,i), need U[i, :] and V[:, k]
    
    # Vectorized verification
    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        i_vals = np.repeat(np.arange(n), m*p)
        j_vals = np.tile(np.repeat(np.arange(m), p), n)
        k_vals = np.tile(np.arange(p), n*m)
        
        u_idx = i_vals * m + j_vals
        v_idx = j_vals * p + k_vals
        w_idx = k_vals * n + i_vals
        
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        
        products = U_sel * V_sel * W_sel
        sums = products.sum(axis=1)
        
        return np.all(np.abs(sums - 1) < tol)

    # Solve for a given rank using alternating least squares
    def solve_rank(r, num_trials, max_iters, tol=1e-10):
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # M[k, r] = V[j, r] * W[k, r]
                        M = V[j*p:(j+1)*p, :] * W[:, i]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # M[i, r] = U[i, r] * W[k, r]
                        M = U[:, j] * W[k*n:(k+1)*n, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # M[j, r] = U[i, r] * V[j, r]
                        M = U[i*m:(i+1)*m, :] * V[:, k]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 0 or it == max_iters - 1:
                    if verify(U, V, W, tol):
                        return U, V, W
        
        return None

    # Try ranks from low to high
    # Lower bound for all-ones tensor J_{n,m,p} is n+m+p-2
    # For 3,3,7: 3+3+7-2 = 11
    # Start from 11 to save time on infeasible 10
    
    # Allocate more trials/iterations for lower ranks as they are harder
    rank_configs = {
        11: (15, 200),
        12: (15, 200),
        13: (10, 150),
        14: (10, 150),
        15: (5, 100),
        16: (5, 100)
    }
    
    # If rank not in config, use default
    default_config = (5, 100)
    
    for r in range(11, best_rank):
        if time.time() > deadline - 10:
            break
            
        if r in rank_configs:
            num_trials, max_iters = rank_configs[r]
        else:
            num_trials, max_iters = default_config
            
        result = solve_rank(r, num_trials, max_iters)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
