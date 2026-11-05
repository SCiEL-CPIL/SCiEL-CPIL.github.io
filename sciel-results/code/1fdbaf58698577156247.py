"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    # Convert to float for matrix operations
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    # Degrees
    D = A.sum(axis=1)
    
    # Inverse degrees (shifted to avoid division by zero)
    inv_D = 1.0 / (D + 1.0)
    
    # Sum of inverse degrees of neighbors
    neighbor_inv_sum = A @ inv_D
    
    # Score:
    # 1. -D: Prefer low degree vertices.
    # 2. -neighbor_inv_sum: Penalize vertices that have neighbors with low degrees.
    #    If a vertex v has neighbors with low degrees, those neighbors are "valuable" 
    #    (easy to include) and we don't want to block them by picking v.
    #    Thus, v should have a lower priority.
    scores = -D - neighbor_inv_sum
    
    # Safety check (though this is very fast)
    if time.time() - start_time > 110:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
