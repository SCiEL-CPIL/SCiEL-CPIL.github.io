"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inverses = {}
    for x in range(1, 11):
        inverses[x] = pow(x, -1, 101)
    
    # For each residue, find the minimum size subset that sums to it
    # Then use bipartite matching to assign subsets to residues minimizing max size
    
    # First, compute for each residue what sizes are achievable
    # There are only 1023 non-empty subsets, so we can enumerate all
    
    # Build list of all subsets with their residue and size
    all_subsets = []  # (residue, size, mask)
    for mask in range(1, 1 << 10):
        terms = []
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                terms.append(x)
        residue = sum(inverses[x] for x in terms) % 101
        size = len(terms)
        all_subsets.append((residue, size, mask, terms))
    
    # For each residue, group subsets by size
    from collections import defaultdict
    subsets_by_residue_size = defaultdict(lambda: defaultdict(list))
    for residue, size, mask, terms in all_subsets:
        subsets_by_residue_size[residue][size].append((mask, terms))
    
    # Find the minimum size needed for each residue
    min_size_for_residue = {}
    for r in range(101):
        for s in range(1, 11):
            if subsets_by_residue_size[r][s]:
                min_size_for_residue[r] = s
                break
    
    # Now we need to assign to each residue a subset such that:
    # 1. Each residue gets exactly one subset
    # 2. No two residues get the same subset (since denominators can't repeat within a row,
    #    but different rows can use same denominators - wait, re-reading the task)
    # 
    # Actually re-reading: "Denominators may recur across different rows, but never within one row."
    # So different rows CAN use the same denominators. This means we don't need a matching!
    # Each row just needs ONE subset whose sum is the residue.
    # We want to minimize the largest row length.
    #
    # So for each residue, we just pick the smallest subset that sums to it.
    # The answer is max over all residues of min_size_for_residue[r].
    
    # Let's verify: can every residue be represented?
    # The seed program shows it can (it uses setdefault which finds at least one per residue)
    
    max_min_size = 0
    best_assignment = {}
    for r in range(101):
        min_sz = min_size_for_residue[r]
        max_min_size = max(max_min_size, min_sz)
        # Pick the first subset of minimum size
        best_assignment[r] = subsets_by_residue_size[r][min_sz][0][1]  # terms
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return what we have
        return [best_assignment[r] for r in range(101)]
    
    # Try to see if we can do better with a more careful search
    # The current approach already finds the theoretical minimum for each residue independently
    # Since rows are independent (denominators can recur), this is optimal.
    
    # But wait - let me double check by trying to find if any residue needs size > some bound
    # and see if we missed any subsets
    
    # Actually, let's verify completeness: all 101 residues must be covered
    missing = [r for r in range(101) if r not in best_assignment]
    if missing:
        # Fallback: use the original approach to ensure coverage
        representations = {}
        for mask in range(1, 1 << 10):
            terms = [x for x in range(1, 11) if mask >> (x - 1) & 1]
            residue = sum(inverses[x] for x in terms) % 101
            if residue not in representations:
                representations[residue] = terms
        return [representations[r] for r in range(101)]
    
    return [best_assignment[r] for r in range(101)]
# EVOLVE-BLOCK-END
