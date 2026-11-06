import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute inverses of 1..10 modulo 101
    invs = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all possible subsets and their sums
    # subset_mask -> (sum_mod_101, denominators_list, size)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x + 1 for x in range(10) if (mask >> x) & 1]
        s = sum(invs[x] for x in range(10) if (mask >> x) & 1) % 101
        size = len(terms)
        all_subsets.append((s, terms, size))
    
    # For each residue, collect all subsets that produce it, grouped by size
    residue_subsets = {r: [] for r in range(101)}
    for s, terms, size in all_subsets:
        residue_subsets[s].append((size, terms))
    
    # Sort by size for each residue
    for r in range(101):
        residue_subsets[r].sort(key=lambda x: x[0])
    
    # Binary search for minimum K
    # Check if all residues can be covered with max subset size K
    def can_cover(K):
        """Check if every residue has at least one subset of size <= K"""
        for r in range(101):
            if not residue_subsets[r]:
                return False
            # Check if any subset has size <= K
            if residue_subsets[r][0][0] > K:
                return False
        return True
    
    # Find minimum K
    min_K = 1
    max_K = 10
    while min_K < max_K:
        mid = (min_K + max_K) // 2
        if can_cover(mid):
            max_K = mid
        else:
            min_K = mid + 1
    
    K = min_K
    
    # Now use Algorithm X (Dancing Links) to find an exact cover
    # Columns: residues 0..100 (each must be covered exactly once)
    # Rows: for each residue r, each subset of size <= K that sums to r
    # We need to select one row per column (residue)
    
    # Build the exact cover matrix
    # columns = 101 residues
    # For each residue r, the available rows are subsets of size <= K summing to r
    
    # Since each column (residue) can only be covered by rows that target that residue,
    # and each row covers exactly one column, this is actually a simple assignment:
    # For each residue r, pick one subset from residue_subsets[r] with size <= K
    
    # But we also need to minimize the maximum row length!
    # Wait - the objective is to minimize the largest row length across all 101 rows.
    # Since we're already constraining to size <= K, and K is the minimum such that coverage is possible,
    # we just need to pick for each residue a subset of size <= K.
    # The maximum row length will be at most K.
    
    # To minimize the maximum, we should try to use smaller subsets where possible.
    # Greedy: for each residue, pick the smallest subset available (size <= K)
    
    result = [None] * 101
    for r in range(101):
        # Pick the smallest subset of size <= K
        for size, terms in residue_subsets[r]:
            if size <= K:
                result[r] = terms
                break
        if result[r] is None:
            # Should not happen if can_cover(K) is True
            result[r] = residue_subsets[r][0][1]
    
    # Verify the solution
    for r in range(101):
        if not result[r]:
            raise ValueError(f"Residue {r} not covered")
        s = sum(invs[x-1] for x in result[r]) % 101
        if s != r:
            raise ValueError(f"Residue {r}: got sum {s}, expected {r}")
        if len(set(result[r])) != len(result[r]):
            raise ValueError(f"Residue {r}: duplicate denominators")
        if any(x < 1 or x > 10 for x in result[r]):
            raise ValueError(f"Residue {r}: denominator out of range")
    
    # Check deadline
    if time.time() > deadline:
        pass  # We already have a valid solution
    
    return result
