def construct():
    import itertools
    
    points = list(range(12))
    all_5_subsets = list(itertools.combinations(points, 5))
    all_6_subsets = list(itertools.combinations(points, 6))
    
    # Map 5-subsets to indices
    subset_index = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Precompute block masks and reverse mapping
    # block_masks[i] is an integer bitmask of the 5-subsets covered by all_6_subsets[i]
    block_masks = [0] * len(all_6_subsets)
    subset_to_blocks = [[] for _ in all_5_subsets]
    
    for b_idx, block in enumerate(all_6_subsets):
        mask = 0
        for i in range(6):
            # Get the 5-subset by removing the i-th element
            # block is a tuple of 6 elements
            sub = tuple(x for j, x in enumerate(block) if j != i)
            idx = subset_index[sub]
            mask |= (1 << idx)
            subset_to_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    full_mask = (1 << len(all_5_subsets)) - 1
    
    import sys
    sys.setrecursionlimit(10000)
    
    def solve(covered_mask, depth, path):
        if depth == 132:
            if covered_mask == full_mask:
                return path
            return None
            
        # Find the first uncovered 5-subset
        # Optimization: We can iterate through bits
        # But simple loop is okay for 792
        # Actually, we can find the lowest set bit of ~covered_mask?
        # Python doesn't have fast bit_count or ctz.
        
        # Just find first i where bit is 0
        # We can use a trick:
        # uncovered = full_mask ^ covered_mask
        # If uncovered == 0: return None (or path if depth==132)
        
        uncovered = full_mask ^ covered_mask
        if uncovered == 0:
            return path if depth == 132 else None
            
        # Find index of lowest set bit in uncovered
        # i = (uncovered & -uncovered).bit_length() - 1
        lsb = uncovered & -uncovered
        i = lsb.bit_length() - 1
        
        cands = subset_to_blocks[i]
        
        for b_idx in cands:
            if covered_mask & block_masks[b_idx] == 0:
                res = solve(covered_mask | block_masks[b_idx], depth + 1, path + [all_6_subsets[b_idx]])
                if res:
                    return res
        return None

    result = solve(0, 0, [])
    
    if result is None:
        # Fallback to greedy or return empty
        return []
        
    return [list(b) for b in result]
