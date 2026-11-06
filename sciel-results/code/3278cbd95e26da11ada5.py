"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
import time as _time

def construct():
    _t0 = _time.time()
    _DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 101
    DENOMS = list(range(1, 11))
    INVS = {x: pow(x, -1, MOD) for x in DENOMS}
    
    # Precompute all non-empty subset sums (2^10 - 1 = 1023 subsets)
    # For each residue, track the minimum-size subset that produces it
    # Since denominators can recur across rows, the problem is separable:
    # each residue independently picks its smallest valid subset.
    
    min_size = [11] * MOD  # 11 = impossible sentinel
    min_terms = [None] * MOD
    
    # Precompute all subsets and their properties
    subsets_by_size = defaultdict(list)  # size -> list of (residue, terms)
    
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        for i in range(10):
            if mask & (1 << i):
                x = i + 1
                terms.append(x)
                res = (res + INVS[x]) % MOD
        size = len(terms)
        subsets_by_size[size].append((res, terms))
    
    # For each residue, find minimum size subset
    for size in range(1, 11):
        for res, terms in subsets_by_size[size]:
            if size < min_size[res]:
                min_size[res] = size
                min_terms[res] = terms
    
    # Check if all residues are covered
    uncovered = [r for r in range(MOD) if min_terms[r] is None]
    
    if uncovered:
        # Fallback: try to cover uncovered residues with any available subset
        # This shouldn't happen, but guard against it
        all_subsets = []
        for size in range(1, 11):
            all_subsets.extend(subsets_by_size[size])
        
        for r in uncovered:
            # Find any subset that produces residue r
            for res, terms in all_subsets:
                if res == r:
                    min_terms[r] = terms
                    min_size[r] = len(terms)
                    break
    
    # Verify solution
    result = []
    for r in range(MOD):
        if min_terms[r] is None:
            result.append([])
        else:
            result.append(min_terms[r])
    
    return result

# EVOLVE-BLOCK-END
