"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # Generate all non-empty subsets (masks) of {1..10}
    # There are 2^10 - 1 = 1023 such subsets
    masks = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        popcount = bin(mask).count('1')
        masks.append((residue, popcount, terms))
    
    # We need to assign each residue 0..100 to a mask such that:
    # 1. Every residue is covered
    # 2. Minimize the maximum popcount (row length)
    # 
    # Strategy: Binary search on the maximum allowed length L.
    # For a given L, check if we can cover all 101 residues using only
    # masks with popcount <= L. Since denominators can recur across rows,
    # we just need: for each residue r, there exists at least one mask
    # with residue r and popcount <= L.
    #
    # Wait - the problem says "Minimize the largest row length."
    # Each row r must contain distinct denominators whose inverses sum to r.
    # Different rows can reuse denominators. So we just need, for each residue r,
    # to find ANY subset of {1..10} whose inverse-sum mod 101 equals r.
    # The cost for residue r is the minimum popcount among all subsets that sum to r.
    # The answer is max over all r of (min popcount for r).
    #
    # This is much simpler than a flow problem! We just need:
    # For each residue r in 0..100, find the minimum-size subset of {1..10}
    # whose modular inverse sum equals r mod 101.
    # Then the answer is the maximum of these minimum sizes.
    
    # For each residue, track the minimum popcount seen
    min_popcount = [float('inf')] * 101
    best_terms = [None] * 101
    
    for residue, popcount, terms in masks:
        if popcount < min_popcount[residue]:
            min_popcount[residue] = popcount
            best_terms[residue] = terms
    
    # Check that all residues are covered
    for r in range(101):
        if best_terms[r] is None:
            raise ValueError(f"Residue {r} not covered")
    
    # The result is a list of 101 lists, row r contains the terms
    # To minimize the largest row length, we use the minimum-popcount subset for each residue
    result = [best_terms[r] for r in range(101)]
    
    return result
# EVOLVE-BLOCK-END
