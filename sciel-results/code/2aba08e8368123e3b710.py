"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Try to find a lower rank decomposition
    # The tensor for matrix multiplication C = AB has entries:
    # T[i,j,k,l] = delta_{j,k} (where i is row of C, l is col of C, j is col of A/row of B, k is row of B/col of A)
    # Rank-1 term: U[i,j,r] * V[j,k,r] * W[k,l,r] should sum over r to delta_{j,k}
    
    # For this specific case, let's try a structured approach
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,l,r] = delta_{j,k} for all i,j,k,l
    
    # This is equivalent to: for each r, we have matrices U_r (n×m), V_r (m×p), W_r (p×n)
    # such that sum_r U_r ⊗ V_r ⊗ W_r = multiplication tensor
    
    # Let's try to see if we can use a lower rank
    # Try ranks from 41 down to see if we can find a valid decomposition
    
    # For now, return the trivial solution as it's guaranteed to work
    return best_result
# EVOLVE-BLOCK-END
