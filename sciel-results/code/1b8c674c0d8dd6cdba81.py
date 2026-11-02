"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Target: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    # Rank-r decomposition: T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    
    # Nonzero entries
    nz = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    
    np.random.seed(12345)
    
    # Try ranks from lower bound up
    lb = max(n*m, m*p, p*n)
    
    for r in range(lb, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Block coordinate descent: iterate fixing two, solving for third
            converged = False
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U: for each (a,b,c) in nz, sum_t U[a,t]*V[b,t]*W[c,t] = 1
                # For each a, the equations are: sum_t U[a,t] * (V[b,t]*W[c,t]) = 1 for each (b,c) with a=i*m+j,b=j*p+k,c=k*n+i
                # Also need T[a,b,c]=0 for all other (b,c)
                
                # Build linear system for U
                # Variables: U[a,t] for t=0..r-1, for each a
                # Equations: for each (a,b,c): sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
                
                # This is a big system. Let me use a simpler approach:
                # Use gradient descent on the loss function
                pass
            
            # Check if we got a good solution (this is a placeholder)
            # For now, skip to next trial
    
    return best
# EVOLVE-BLOCK-END
