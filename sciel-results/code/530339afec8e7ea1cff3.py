import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-35 decomposition
    rank = 35
    
    # Set up the tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try a numerical search
    best_rank = 36
    best_decomp = None
    
    # Initialize with the known rank-36 decomposition
    # ... (same as before)
    
    # Try to reduce to rank 35
    # ... (numerical search)
    
    # If search fails, return rank-36
    # ...
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
