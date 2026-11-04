import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Fallback: Schoolbook
    def get_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try rank 74
    target_rank = 74
    start_time = time.time()
    max_search_time = 30.0
    
    # Initialize
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # Target tensor T[i,j,k] = 1
    # We want sum_r U[i,r] V[j,r] W[k,r] = 1
    
    # ALS Iterations
    for it in range(1000):
        if time.time() - start_time > max_search_time:
            break
            
        # Update W
        # For each k, we have equations for all i,j: sum_r (U[i,r] V[j,r]) W[k,r] = 1
        # This is a linear system A W_k = b
        # A is (n*m) x target_rank
        # A[i*m+j, r] = U[i,r] * V[j,r]
        # b[i*m+j] = 1
        A = np.zeros((n*m, target_rank))
        for i in range(n):
            for j in range(m):
                A[i*m+j, :] = U[i, :] * V[j, :]
        b = np.ones(n*m)
        # Solve A W_col = b for each k?
        # No, W has shape (p*n, target_rank).
        # The index in W is k*n + i.
        # The equation is sum_r (U[i,r] V[j,r]) W[k*n+i, r] = 1.
        # This couples i and k in W.
        # This is not a simple block diagonal solve for W.
        # It's a general linear system for all W variables.
        # Variables: W_{k*n+i, r}
        # Equation for (i,j,k): sum_r (U[i,r] V[j,r]) W_{k*n+i, r} = 1
        
        # Let's flatten W to vector w of size p*n*target_rank
        # Coefficient for w[index] where index = (k*n+i)*target_rank + r is (U[i,r] V[j,r])
        
        # This system is large: n*m*p equations = 75.
        # Variables: p*n*target_rank = 25 * 74 = 1850.
        # We can use np.linalg.lstsq.
        
        # Construct matrix M (75 x 1850)
        # Row idx = i*m*p + j*p + k ? No, i,j,k order.
        # Let's just build it.
        M = np.zeros((n*m*p, p*n*target_rank))
        b_full = np.ones(n*m*p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Equation idx: sum_r U[i,r] V[j,r] W[k*n+i, r] = 1
                    for r in range(target_rank):
                        # W variable index: (k*n + i) * target_rank + r
                        w_idx = (k*n + i) * target_rank + r
                        M[idx, w_idx] = U[i, r] * V[j, r]
                    idx += 1
        
        # Solve M @ w_flat = b_full
        w_sol, residuals, rank_M, s = np.linalg.lstsq(M, b_full, rcond=None)
        W = w_sol.reshape(p*n, target_rank)
        
        # Check error
        # Recalculate tensor
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k*n+i, r]
        # This is getting slow to check every time.
        # Only check every 10 iterations.
        
        if it % 10 == 0:
             # Quick check
             err = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         val = 0
                         for r in range(target_rank):
                             val += U[i,r] * V[j,r] * W[k*n+i, r]
                         err += (val - 1)**2
             if err < 1e-10:
                 # Found a solution?
                 # Need to check if it can be represented as half-integers
                 # And verify exactly.
                 pass
                 
    return get_schoolbook()
