"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s wall-clock kill
    
    # Precompute modular inverses of 1..10 mod 101
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # Precompute all subsets grouped by size and residue
    # subsets_by_size[size][residue] = list of subsets of that size achieving that residue
    subsets_by_size = [[] for _ in range(11)]  # index 1..10
    for size in range(1, 11):
        subsets_by_size[size] = [[] for _ in range(p)]
        for comb in combinations(range(1, 11), size):
            s = sum(inv[x] for x in comb) % p
            subsets_by_size[size][s].append(list(comb))
    
    # Binary search on max_size: find smallest max_size such that all residues can be covered
    # using only subsets of size <= max_size
    # For a given max_size, we need to assign each residue a subset from sizes 1..max_size
    # such that each residue gets exactly one subset.
    # Since subsets can be reused across residues, this is always possible if for every residue
    # there exists at least one subset of size <= max_size achieving it.
    # But we want to minimize the MAXIMUM row length, so we want the smallest max_size
    # where every residue has at least one subset of size <= max_size.
    
    # First, find for each residue the minimum size needed
    min_size_for_residue = [11] * p  # 11 means "not found yet"
    for size in range(1, 11):
        for r in range(p):
            if min_size_for_residue[r] == 11 and len(subsets_by_size[size][r]) > 0:
                min_size_for_residue[r] = size
    
    # The answer is max(min_size_for_residue[r] for r in 0..100)
    # But we can do better: we don't need each residue to use its minimum size.
    # We need to find the smallest M such that we can assign to each residue r a subset
    # of size <= M. This is equivalent to: for each residue r, there exists a subset
    # of size <= M achieving r. So the answer is max_r (min size of subset achieving r).
    
    max_needed = max(min_size_for_residue)
    
    # Now, for each residue, pick a subset of size <= max_needed.
    # To minimize the maximum, we already have max_needed as the upper bound.
    # But we can try to see if we can do better with a smarter assignment.
    # Actually, the minimum possible maximum is exactly max_r (min size for r),
    # because each residue r MUST use a subset of size at least min_size_for_residue[r].
    
    # So the optimal maximum row length is max_needed.
    # Now assign each residue a subset of size <= max_needed (prefer smaller for tie-breaking).
    best = [None] * p
    for r in range(p):
        # Find the smallest size subset for this residue that is <= max_needed
        assigned = None
        for size in range(1, max_needed + 1):
            if len(subsets_by_size[size][r]) > 0:
                assigned = subsets_by_size[size][r][0]
                break
        if assigned is None:
            # Should not happen if max_needed is computed correctly
            for size in range(1, 11):
                if len(subsets_by_size[size][r]) > 0:
                    assigned = subsets_by_size[size][r][0]
                    break
        best[r] = assigned
    
    # Verify: check that all rows are nonempty and valid
    for r in range(p):
        if best[r] is None or len(best[r]) == 0:
            # Fallback: use any subset
            for size in range(1, 11):
                if len(subsets_by_size[size][r]) > 0:
                    best[r] = subsets_by_size[size][r][0]
                    break
    
    return [best[r] for r in range(p)]
# EVOLVE-BLOCK-END
