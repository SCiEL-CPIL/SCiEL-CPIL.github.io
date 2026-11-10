"""Combined strategy: Maximize pairwise disjoint residue classes using the property that 
equal residues on pairwise non-coprime moduli are disjoint.
The largest set of pairwise non-coprime moduli in [2, 128] is the set of even numbers."""

import time
import math

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    d = abs(r1 - r2)
    return math.gcd(math.gcd(d, m1), m2) > 1

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin
    MAX_MOD = 128
    
    # The optimal strategy is to select the largest subset of moduli 
    # that are pairwise non-coprime, and assign the same residue to all.
    # The largest such subset in [2, MAX_MOD] is the set of even numbers.
    # Size is MAX_MOD // 2 = 64.
    
    # Verify this logic holds and construct the result.
    # Moduli: 2, 4, ..., 128
    # Residue: 1 (or any fixed value)
    
    result = []
    for m in range(2, MAX_MOD + 1, 2):
        result.append([1, m])
        
    # Verification (optional, but good for safety to ensure no bug in logic)
    # We know mathematically it works, but let's double check a few random pairs 
    # or all pairs if fast enough (64 choose 2 = 2016 checks, very fast).
    
    valid = True
    for i in range(len(result)):
        r1, m1 = result[i]
        for j in range(i+1, len(result)):
            r2, m2 = result[j]
            if not _is_disjoint(r1, m1, r2, m2):
                valid = False
                break
        if not valid:
            break
            
    if valid:
        return result
    else:
        # Fallback to seed if logic fails (should not happen)
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]

# EVOLVE-BLOCK-END
