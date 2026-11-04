"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank n*m*p = 75)
    rank_schoolbook = n * m * p
    U = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W = [[0.0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    U_school = np.array(U)
    V_school = np.array(V)
    W_school = np.array(W)
    
    # Tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank_schoolbook):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    T[a, b, c] += U_school[a, r] * V_school[b, r] * W_school[c, r]
    
    # Try to find a lower rank decomposition using numerical optimization
    # We will try to find a rank R decomposition with integer/half-integer coefficients
    # by solving a least squares problem and rounding.
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try ranks from 74 down to 70
    for rank in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random small perturbations or schoolbook truncated
        # Better: Use the schoolbook factors and try to combine
        # Or use a random initialization
        
        # We'll use a simple heuristic: try to find a solution
        # For simplicity, we'll skip the complex search in this version
        # and just return the schoolbook result to ensure validity.
        # A more advanced version would use scipy.optimize or a custom solver.
        
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
