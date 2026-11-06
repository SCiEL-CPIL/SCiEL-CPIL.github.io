def construct():
    import time
    from collections import defaultdict
    
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: mask -> (size, residue, list_of_denominators)
    # There are 1023 non-empty subsets
    subsets = []  # list of (size, residue, denominators_list)
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        residue = 0
        dens = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                residue = (residue + inv[x - 1]) % 101
                dens.append(x)
        subsets.append((size, residue, dens))
    
    # Group subsets by residue
    by_residue = defaultdict(list)
    for i, (size, residue, dens) in enumerate(subsets):
        by_residue[residue].append(i)
    
    # For each residue, sort subsets by size (prefer smaller)
    for r in by_residue:
        by_residue[r].sort(key=lambda i: subsets[i][0])
    
    # Binary search on K: minimum max size K such that we can assign each residue
    # a distinct subset of size <= K
    # Note: subsets are shared across residues (a subset has exactly one residue),
    # so actually each subset belongs to exactly one residue. This means the problem
    # is: for each residue r, pick one subset from by_residue[r] such that all chosen
    # subsets are distinct (they automatically are since each subset has unique residue).
    # Wait - each subset maps to exactly one residue, so subsets in different residue
    # groups are automatically distinct! The constraint "denominators may recur across
    # different rows, but never within one row" is already satisfied by each subset.
    # So the problem reduces to: for each residue, pick the smallest subset that covers it.
    # Since each subset has a unique residue, there's no conflict between residues.
    
    # Actually wait - let me re-read. "row r containing distinct denominators in 1..10 
    # whose modular inverses sum to r modulo 101". Each row is a set of denominators.
    # Two different rows can share denominators. The only constraint is within a row,
    # denominators are distinct (which is automatic for a subset).
    # So for each residue r, we just need to find ANY non-empty subset of {1..10} 
    # whose inverse sum is r mod 101, and we want to minimize the maximum size across all 101 rows.
    # Since subsets are uniquely determined by their element set, and each element set 
    # has exactly one residue, there's no overlap issue. We just need for each residue,
    # the minimum-size subset that achieves it.
    
    # But wait - can two different subsets have the same residue? Yes! Multiple subsets
    # can sum to the same residue mod 101. But we only need ONE per residue.
    # And since we're picking independently for each residue, there's no conflict.
    
    # So the answer is simply: for each residue, find the minimum size subset.
    # The objective is max over all residues of that minimum size.
    
    # Let's compute the minimum size for each residue
    min_size = [11] * 101
    best_subset = [None] * 101
    
    for i, (size, residue, dens) in enumerate(subsets):
        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = dens
    
    # Check all residues are covered
    assert all(s is not None for s in best_subset), "Not all residues covered"
    
    return best_subset
