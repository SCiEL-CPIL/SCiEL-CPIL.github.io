"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
# Large prime for modular pre-filter
MOD_P = 10**9 + 7
# Precompute modular inverses of 1..22 modulo MOD_P
_MOD_INV = [0] * 23
for _i in range(1, 23):
    _MOD_INV[_i] = pow(_i, MOD_P - 2, MOD_P)

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    # Fallback to seed if search fails or times out
    # Seed is size 9
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Try to find larger sets
    # We iterate k from 22 down to 10
    # If we find a valid set of size k, we return it immediately
    # because we are searching in descending order of k
    
    for k in range(22, 9, -1):
        # Generate combinations in lexicographic order
        # This prioritizes sets with smaller numbers (larger weights),
        # which are more likely to be dissociated.
        for comb in combinations(range(1, 23), k):
            # Time check
            if time.time() - start_time > deadline:
                return best_set
            
            # Fast modular pre-filter
            if not _modular_dissociated(comb):
                continue
            
            # Exact check
            if _is_dissociated(comb):
                return list(comb)
        
        # If no valid set found for this k, continue to next k
    
    return best_set

def _modular_dissociated(S):
    """Fast necessary condition: all subset sums of modular inverses must be distinct mod P."""
    weights = [_MOD_INV[x] for x in S]
    
    sums = {0}
    for w in weights:
        new_sums = {(s + w) % MOD_P for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
    
    return True

def _is_dissociated(S):
    # Calculate LCM
    L = 1
    for x in S:
        L = L * x // math.gcd(L, x)
    
    # Calculate weights
    weights = [L // x for x in S]
    
    # Check if all subset sums are distinct
    # We use a set to store sums. If a sum is repeated, it's not dissociated.
    sums = {0}
    for w in weights:
        # Generate new sums by adding w to all existing sums
        # We can do this efficiently by creating a set of new sums
        new_sums = {s + w for s in sums}
        
        # Check for intersection with existing sums
        # If intersection is not empty, a collision occurred
        if new_sums & sums:
            return False
        
        sums |= new_sums
    
    return True
# EVOLVE-BLOCK-END
