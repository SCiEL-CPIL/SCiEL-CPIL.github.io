"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i,j,k] = 1 if the (i,k) output element depends on A[i,j] and B[j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Try to find lower rank decompositions
    best_rank = schoolbook_rank
    best_factors = None
    
    # Start from schoolbook and try to reduce
    for target_rank in range(max(n*m, m*p, p*n), schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            
            # Initialize factors with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares optimization
            for iteration in range(100):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each row i*m+j, solve least squares
                for i in range(n):
                    for j in range(m):
                        # T[i,j,:] = sum_alpha U[i*m+j, alpha] * (V[:,alpha] * W[:,alpha])_k
                        # This is complex; use gradient-based instead
                        pass
                
                # Instead, use a simpler approach: direct tensor decomposition
                # T[i,j,k] = sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha]
                
                # This is a non-convex problem. Let's try a different strategy.
                break
            
            # Check if this random initialization gives a valid decomposition
            # (unlikely, but check for completeness)
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        error += (val - T[i,j,k]) ** 2
            
            if error < 1e-10:
                best_rank = target_rank
                best_factors = (U, V, W)
                break
        
        if best_factors is not None:
            break
    
    # If no better found, return schoolbook
    if best_factors is None:
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_factors
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
