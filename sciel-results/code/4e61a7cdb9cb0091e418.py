"""Search-based program: find minimum max row length covering all residues mod 101."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # Build all nonempty subsets with their residues, grouped by size
    subsets_by_size = {i: [] for i in range(1, 11)}
    
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        for x in D:
            if mask & (1 << (x - 1)):
                terms.append(x)
                res = (res + inv[x]) % P
        size = len(terms)
        subsets_by_size[size].append((res, terms))
    
    # Find minimum k such that all 101 residues are covered by subsets of size <= k
    for k in range(1, 11):
        if time.time() - t0 > deadline - 5:
            break
        covered = {}
        for size in range(1, k + 1):
            for res, terms in subsets_by_size[size]:
                if res not in covered:
                    covered[res] = terms
        
        if len(covered) == 101:
            # Valid covering found with max row length k
            result = [covered[r] for r in range(101)]
            return result
    
    # Fallback: use all subsets
    covered = {}
    for size in range(1, 11):
        for res, terms in subsets_by_size[size]:
            if res not in covered:
                covered[res] = terms
    
    if len(covered) == 101:
        return [covered[r] for r in range(101)]
    
    # Last resort: original approach
    representations = {}
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if mask >> (x - 1) & 1]
        residue = sum(inv[x] for x in terms) % 101
        representations.setdefault(residue, terms)
    return [representations[r] for r in range(101)]

# EVOLVE-BLOCK-END
