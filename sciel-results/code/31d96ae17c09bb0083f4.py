"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook as baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition via iterative refinement
    # The tensor is T[i,j,k] = 1 for the matrix multiplication
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Use a simple iterative approach: try random initializations and gradient-free optimization
    # Given time constraints, try a few random rank-1 reductions
    
    # Strategy: try to reduce rank by finding linear dependencies
    # For now, return schoolbook as it's guaranteed correct
    # A more sophisticated approach would use algebraic methods
    
    # Try a few random perturbations to see if we can verify a lower rank
    # But without a proper solver, this is risky. Stick with schoolbook for correctness.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
