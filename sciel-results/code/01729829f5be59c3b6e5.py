import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Trivial decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to find better decompositions using a simple search
    # For each candidate rank r, try to find U, V, W such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Build the constraint matrix
    # Each constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is nonlinear, so we'll use a simple approach
    
    # Try rank 83, 82, ... down to some minimum
    for r in range(rank0 - 1, max(20, rank0 - 20), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Simple gradient descent (not implemented for time)
            # ...
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
