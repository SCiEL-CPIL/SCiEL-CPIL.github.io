import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def priority(vector, n):
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Return a neutral value; this should rarely trigger
        # since priority is called per-vector in a loop
        return 0.0
    
    # Count occurrences of 0, 1, 2
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Ideal count is n/3
    ideal = n / 3.0
    
    # Deviation from ideal balance (lower is better)
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Smooth structural penalty: penalize pairs of coordinates that are
    # "close" in the F_3 sense (differ by 0 or 1 mod 3), which increases
    # collision risk with other vectors. Use a soft-count via exp(-beta*d)
    # where d is the cyclic distance in {0,1,2}.
    # For n up to ~12, O(n^2) is fine.
    if n <= 20:
        structural = 0.0
        beta = 0.8
        for i in range(n):
            vi = vector[i]
            for j in range(i + 1, n):
                vj = vector[j]
                diff = (vi - vj) % 3
                # cyclic distance: min(diff, 3-diff) -> 0,1,1
                dist = min(diff, 3 - diff)
                structural += math.exp(-beta * dist)
        # Normalize by number of pairs
        npairs = n * (n - 1) / 2.0
        structural /= npairs if npairs > 0 else 1.0
    else:
        structural = 0.0
    
    # Deterministic polynomial hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val) % 1000000007
    
    # Normalize hash to [0, 1)
    tie_breaker = h / 1000000007.0
    
    # Combine: primary balance, secondary structural repulsion, tertiary tie-break
    # structural ranges roughly [0, 1] (1 when all coords equal, ~0.5 when mixed)
    # We want to PENALIZE high structural similarity, so subtract it.
    return -dev - 2.0 * structural + 0.01 * tie_breaker
