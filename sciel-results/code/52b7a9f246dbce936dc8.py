"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    MOD = 101
    DENOMS = list(range(1, 11))
    INVS = {x: pow(x, -1, MOD) for x in DENOMS}
    
    # Precompute all non-empty subset sums (2^10 - 1 = 1023 subsets)
    # For each residue, track the minimum-size subset that produces it
    # Since denominators can recur across rows, the problem is separable:
    # each residue independently picks its smallest valid subset.
    
    # min_size[r] = smallest subset size producing residue r
    # min_terms[r] = the corresponding terms
    min_size = [11] * MOD  # 11 = impossible sentinel
    min_terms = [None] * MOD
    
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        for i in range(10):
            if mask & (1 << i):
                x = i + 1
                terms.append(x)
                res = (res + INVS[x]) % MOD
        size = len(terms)
        if size < min_size[res]:
            min_size[res] = size
            min_terms[res] = terms
    
    # Build result
    result = []
    for r in range(MOD):
        if min_terms[r] is None:
            # Should not happen, but handle gracefully
            result.append([])
        else:
            result.append(min_terms[r])
    
    return result

# EVOLVE-BLOCK-END
