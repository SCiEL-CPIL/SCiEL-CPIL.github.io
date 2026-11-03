"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Greedy search for a large Sidon set (B2 set)
    # A Sidon set ensures all pairwise sums i+j (i<=j) are distinct.
    # This minimizes the autoconvolution peak relative to the L2 norm.
    
    S = [0]
    used_sums = {0}
    
    # Try to add elements from 1 to N-1
    for x in range(1, N):
        # Calculate new sums created by adding x
        # Sums are x + s for s in S, and x + x
        new_sums = [x + s for s in S]
        new_sums.append(2 * x)
        
        # Check if any new sum collides with an existing sum
        # Note: new_sums are distinct among themselves if S is a Sidon set and x not in S
        collision = False
        for s in new_sums:
            if s in used_sums:
                collision = True
                break
        
        if not collision:
            S.append(x)
            for s in new_sums:
                used_sums.add(s)
            
            # Time guard
            if time.time() - start_time > 110:
                break
    
    k = len(S)
    
    # Construct the height vector
    # For a Sidon set of size k, equal weights yield a score of k/2 (for k>1)
    # which is optimal for this support structure.
    h = np.zeros(N)
    if k > 0:
        val = 1.0 / k
        for idx in S:
            h[idx] = val
    
    return list(h)
# EVOLVE-BLOCK-END
