# EVOLVE-BLOCK-START
def construct():
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # Generate all non-empty subsets (masks) of {1..10}
    # Store as (residue, popcount, terms)
    subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        popcount = len(terms)
        subsets.append((residue, popcount, terms))
    
    # Determine the minimum possible maximum row length L_opt
    # Binary search on L
    # L can range from 1 to 10
    low, high = 1, 10
    L_opt = 10
    
    # Feasibility check for a given L:
    # Can we assign a subset to every residue r such that popcount(subset) <= L?
    # Since denominators can recur, this is equivalent to:
    # For every r, there exists at least one subset with residue r and popcount <= L.
    def is_feasible(L):
        covered = [False] * 101
        for residue, popcount, terms in subsets:
            if popcount <= L:
                covered[residue] = True
        # Check if all residues are covered
        return all(covered)
    
    # Binary search
    while low <= high:
        mid = (low + high) // 2
        if is_feasible(mid):
            L_opt = mid
            high = mid - 1
        else:
            low = mid + 1
            
    # Now, construct the result using subsets with popcount <= L_opt
    # To minimize the "largest row length" further (though it's capped by L_opt),
    # we pick the subset with the minimum popcount for each residue.
    best_terms = [None] * 101
    min_pop_for_res = [float('inf')] * 101
    
    for residue, popcount, terms in subsets:
        if popcount < min_pop_for_res[residue]:
            min_pop_for_res[residue] = popcount
            best_terms[residue] = terms
            
    # Validate
    for r in range(101):
        if best_terms[r] is None:
            raise ValueError(f"Residue {r} not covered")
        # Ensure the chosen subset is within the allowed limit
        # (It should be, since L_opt is the minimum feasible max, and min_pop <= L_opt)
        
    result = [best_terms[r] for r in range(101)]
    return result
# EVOLVE-BLOCK-END
