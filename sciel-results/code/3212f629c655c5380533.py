"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # Generate all non-empty subsets (masks) of {1..10}
    # Store: (residue, popcount, terms)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        popcount = bin(mask).count('1')
        all_subsets.append((residue, popcount, terms))
        
    # Group subsets by residue to allow quick lookup
    # For each residue, we want the subset with the minimum popcount
    # This effectively solves the "minimum cost" assignment where cost is popcount
    # and each residue is independent.
    
    best_for_residue = [None] * 101
    min_size_for_residue = [11] * 101 # Max possible size is 10
    
    for residue, popcount, terms in all_subsets:
        if popcount < min_size_for_residue[residue]:
            min_size_for_residue[residue] = popcount
            best_for_residue[residue] = terms
            
    # Verify coverage
    for r in range(101):
        if best_for_residue[r] is None:
            raise ValueError(f"Residue {r} not covered")
            
    # The optimal solution for minimizing the maximum row length is to pick
    # the minimum-length subset for each residue independently.
    # This is equivalent to finding a perfect matching in a bipartite graph
    # (Residues x Subsets) where edges exist if Sum(S)=r, with edge weight |S|,
    # and minimizing the maximum edge weight in the matching.
    # Due to the structure (each subset belongs to exactly one residue class),
    # the matching is simply choosing the best edge for each residue.
    
    result = [best_for_residue[r] for r in range(101)]
    return result
# EVOLVE-BLOCK-END
