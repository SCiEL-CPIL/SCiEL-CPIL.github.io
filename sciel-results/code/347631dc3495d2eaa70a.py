"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict, deque

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
    
    # Now we need to check if we can assign subsets to residues such that
    # each residue gets a subset of size <= K, and no two residues share the same subset
    # (wait, the problem says denominators may recur across rows, but never within one row)
    # Re-reading: "Denominators may recur across different rows, but never within one row."
    # This means we CAN use the same subset for multiple residues!
    # So the problem IS separable - each residue independently picks its best subset.
    
    # But wait, let me re-read more carefully...
    # "row r containing distinct denominators in 1..10 whose modular inverses sum to r modulo 101"
    # "Denominators may recur across different rows, but never within one row."
    # 
    # This means: within a single row, all denominators must be distinct.
    # Across different rows, the same denominator can appear.
    # 
    # So yes, the problem is separable! Each row independently chooses a subset of {1..10}
    # (with distinct elements) whose inverse sum equals the residue.
    # We just need to minimize the maximum row length.
    
    # The current approach already does this optimally for each residue independently.
    # The maximum row length is max(min_size[r] for r in range(101)).
    
    # Let's verify this is correct and compute the answer
    result = []
    for r in range(MOD):
        if min_terms[r] is None:
            # Should not happen if all residues are covered
            result.append([])
        else:
            result.append(min_terms[r])
    
    return result

# EVOLVE-BLOCK-END
