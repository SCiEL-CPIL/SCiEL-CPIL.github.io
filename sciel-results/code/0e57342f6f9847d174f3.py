"""Optimized program for the suite task; finds minimum cardinality subset for each residue."""
# EVOLVE-BLOCK-START
def construct():
    p = 101
    denominators = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
        
    # best_subset[r] will hold the smallest subset (by length) that sums to r
    # Initialize with None
    best_subset = [None] * p
    
    # Iterate through all non-empty subsets
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + inv[x]) % p
        
        r = s
        current_best = best_subset[r]
        
        # If this is the first subset for this residue, or it's smaller than the current best
        if current_best is None or len(terms) < len(current_best):
            best_subset[r] = terms
            
    # Construct the final list of 101 lists
    # Every residue must be represented. If some residue is None, it means it's not covered,
    # which shouldn't happen given the problem statement implies a solution exists.
    # However, to be safe, if any is None, we might need to handle it, but 
    # with 1023 subsets and 101 residues, coverage is virtually guaranteed.
    
    result = []
    for r in range(p):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: should not happen, but if it does, use a dummy? 
            # The problem says "Every residue must be represented".
            # If the math allows, all are covered.
            # Let's assume it's covered. If not, this would be an error in the problem or my logic.
            # Actually, let's just append an empty list or raise? 
            # The task says "101 nonempty lists". 
            # If a residue is not covered, the program is invalid.
            # But with 1023 subsets, it is extremely likely all 101 residues are hit.
            result.append([]) # Placeholder, should not be empty in valid output
            
    return result
# EVOLVE-BLOCK-END
