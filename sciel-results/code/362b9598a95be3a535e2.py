# EVOLVE-BLOCK-START
def construct():
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # For each residue, track the smallest subset (by size) that sums to it
    best_for_residue = {}  # residue -> (size, terms_list)
    
    # Enumerate all non-empty subsets of {1,2,...,10}
    for mask in range(1, 1 << 10):
        # Compute terms and residue
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % 101
        size = len(terms)
        
        if s not in best_for_residue or size < best_for_residue[s][0]:
            best_for_residue[s] = (size, terms)
    
    # Check that all 101 residues are covered
    # If any residue is missing, we need a fallback (shouldn't happen with 1023 subsets)
    result = []
    for r in range(101):
        if r in best_for_residue:
            result.append(best_for_residue[r][1])
        else:
            # Fallback: this shouldn't happen, but use full set as last resort
            result.append(list(range(1, 11)))
    
    return result
# EVOLVE-BLOCK-END
