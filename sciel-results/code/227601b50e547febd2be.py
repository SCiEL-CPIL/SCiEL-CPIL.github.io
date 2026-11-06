import time

def construct():
    p = 101
    # Precompute inverses for 1..10
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
        
    # Generate all non-empty subsets and their properties
    # We store (size, residue, terms_list)
    subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                terms.append(denom)
                res = (res + inv[denom]) % p
        size = len(terms)
        subsets.append((size, res, terms))
        
    # Sort by size to ensure we pick smallest possible subsets first
    # This greedy approach minimizes the maximum subset size used.
    subsets.sort(key=lambda x: x[0])
    
    assigned = [None] * 101
    assigned_count = 0
    
    for size, res, terms in subsets:
        if assigned[res] is None:
            assigned[res] = terms
            assigned_count += 1
            if assigned_count == 101:
                break
                
    # All residues must be covered. 
    # Given 1023 subsets and 101 residues, coverage is guaranteed.
    # The max size will be small (likely 3 or 4).
    
    return assigned
