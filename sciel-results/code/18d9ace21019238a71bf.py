"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute all non-empty subsets of {1,...,10} with their size and residue
    all_subsets = []  # list of (size, residue, terms)
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        all_subsets.append((size, residue, terms))
    
    # Sort by size to try smaller subsets first
    all_subsets.sort(key=lambda x: x[0])
    
    # Find the minimum k such that all 101 residues are covered by subsets of size <= k
    for k in range(1, 11):
        # Collect all subsets of size <= k
        candidates = [s for s in all_subsets if s[0] <= k]
        # Check if all residues are covered
        covered = set()
        for size, residue, terms in candidates:
            covered.add(residue)
        if len(covered) == 101:
            # Assign to each residue a subset of size <= k
            # Prefer smaller subsets for efficiency
            rep = {}
            for size, residue, terms in candidates:
                if residue not in rep:
                    rep[residue] = terms
            return [rep[r] for r in range(101)]
    
    # Fallback: should not reach here
    rep = {}
    for size, residue, terms in all_subsets:
        if residue not in rep:
            rep[residue] = terms
    return [rep[r] for r in range(101)]
# EVOLVE-BLOCK-END
