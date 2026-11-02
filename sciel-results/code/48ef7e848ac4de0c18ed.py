"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find rank-29 decomposition
    target_rank = 29
    
    # Build the tensor T as a 3D array
    # T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try random initialization and optimization
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        rank = target_rank
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Use iterative refinement
        for iteration in range(200):
            if time.time() > deadline:
                break
                
            # Compute error: E[a,b,c] = T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
            # For each r, update U, V, W
            for r in range(rank):
                # Fix V and W, update U
                # U[a,r] = sum_{b,c} T[a,b,c]*V[b,r]*W[c,r] / sum_{b,c} V[b,r]^2*W[c,r]^2
                # This is a linear system for each r
                
                # Actually, let's use a simpler approach: alternating least squares
                pass
        
        # This is getting too slow. Let me try a different approach.
        break
    
    # Fall back to schoolbook rank-30
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
