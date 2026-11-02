"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Use superincreasing set U = {0, 1, 2, 4, 8, ..., 2^(n-1)}
    # For this set, all pairwise sums and differences are distinct
    # |U-U| = (n+1)^2, |U+U| = (n+1)(n+2)/2
    # ratio = 2(n+1)/(n+2)
    # Score = 1 + log(ratio)/log(2*2^(n-1) + 1)
    
    # Find optimal n by trying values
    best_score = 0
    best_n = 1
    
    for n in range(1, 100):
        # Check time
        if time.time() > deadline:
            break
        
        max_u = 2 ** (n - 1) if n > 1 else 0
        denom = 2 * max_u + 1
        if denom <= 1:
            continue
        
        ratio = 2 * (n + 1) / (n + 2)
        # ratio is at most 2, so log(ratio) <= log(2)
        if ratio <= 1:
            score = 1.0
        else:
            score = 1 + math.log(ratio) / math.log(denom)
        
        if score > best_score:
            best_score = score
            best_n = n
    
    # Build the set
    n = best_n
    U = [0]
    for i in range(1, n + 1):
        U.append(2 ** (i - 1))
    
    return U
# EVOLVE-BLOCK-END
