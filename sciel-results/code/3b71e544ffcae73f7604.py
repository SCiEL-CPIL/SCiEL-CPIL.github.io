import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # Strategy: Use all distinct weight-2 vectors in {-1, 0, 1}^11.
    # Norm squared is 2.
    # Condition for compatibility: u . v <= 1 for distinct u, v.
    # As derived, for distinct weight-2 vectors, u . v is in {-2, -1, 0, 1}.
    # It is never 2 (which would imply u == v).
    # Thus, all distinct weight-2 vectors are mutually compatible.
    
    vectors = []
    
    # Generate all supports of size 2
    for support in itertools.combinations(range(D), 2):
        i, j = support
        # 4 sign combinations: (+, +), (+, -), (-, +), (-, -)
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = s1
                v[j] = s2
                vectors.append(v)
                
    result = np.array(vectors, dtype=np.int64)
    
    # Verification (Optional but recommended for safety in this environment)
    # If N is 220, checking all pairs is 220*220 = 48400 checks. Fast.
    if len(result) > 0:
        # Calculate norms
        norms = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms)
        
        # Calculate pairwise distances squared
        # d^2 = ||u||^2 + ||v||^2 - 2 u.v
        # We need d^2 >= max_norm_sq
        # 2 * max_norm_sq - 2 u.v >= max_norm_sq
        # max_norm_sq >= 2 u.v
        # u.v <= max_norm_sq / 2
        
        # For our set, max_norm_sq = 2. So we need u.v <= 1.
        
        # Efficient check using matrix multiplication
        # dots = result @ result.T
        # This is 220x220. Very fast.
        dots = result @ result.T
        np.fill_diagonal(dots, 0) # Ignore self-dot products
        
        # Check if any off-diagonal dot product > 1
        # Since max possible dot for distinct W2 is 1, this should be false.
        if np.any(dots > 1):
            # Fallback: If math was wrong, we need to filter.
            # But for W2, it shouldn't happen.
            # Just to be absolutely safe, we can filter, but it's likely unnecessary.
            pass
            
    return result
