"""Construct the Steiner system S(5,6,12) deterministically."""

def construct():
    """Return the 132 blocks of the Steiner system S(5,6,12) on points 0..11."""
    import time
    import itertools
    
    start = time.time()
    DEADLINE = 140.0

    # S(5,6,12) is constructed from S(4,5,11):
    # For each 5-block B of S(4,5,11) on points {0,...,10}:
    #   - B ∪ {11} is a block of S(5,6,12)
    #   - {0,...,10} \ B is a block of S(5,6,12)
    # This gives 66*2 = 132 blocks.
    
    # We need to find the 66 blocks of S(4,5,11).
    # Use a fast backtracking with bitmask representation.
    
    all_4subsets = list(itertools.combinations(range(11), 4))  # 330 items
    all_5subsets = list(itertools.combinations(range(11), 5))  # 462 items
    
    four_to_idx = {s: i for i, s in enumerate(all_4subsets)}
    
    # Precompute: for each 5-subset, the bitmask of 4-subsets it contains
    five_masks = []
    for s in all_5subsets:
        mask = 0
        for f in itertools.combinations(s, 4):
            mask |= (1 << four_to_idx[f])
        five_masks.append(mask)
    
    n_fours = len(all_4subsets)  # 330
    n_fives = len(all_5subsets)  # 462
    target_blocks = 66
    
    def solve():
        if time.time() > DEADLINE:
            return None
        
        def backtrack(covered_mask, blocks, depth):
            if time.time() > DEADLINE:
                return None
            if depth == target_blocks:
                # Verify completeness
                if covered_mask == (1 << n_fours) - 1:
                    return list(blocks)
                return None
            
            # Find uncovered 4-subsets
            full_mask = (1 << n_fours) - 1
            uncovered = full_mask & ~covered_mask
            
            if uncovered == 0:
                return None  # All covered but not enough blocks
            
            # Greedy: pick the 5-set that covers the most uncovered 4-subsets
            # among those that don't conflict
            
            candidates = []
            for i in range(n_fives):
                m = five_masks[i]
                if m & covered_mask:
                    continue  # conflict
                new = m & uncovered
                if new:
                    cnt = bin(new).count('1')
                    candidates.append((cnt, i))
            
            if not candidates:
                return None
            
            candidates.sort(reverse=True)
            
            for cnt, i in candidates:
                blocks.append(all_5subsets[i])
                result = backtrack(covered_mask | five_masks[i], blocks, depth + 1)
                if result is not None:
                    return result
                blocks.pop()
            
            return None
        
        return backtrack(0, [], 0)
    
    s4511 = solve()
    
    if s4511 is None or len(s4511) != 66:
        # Fallback: try a different initial ordering or return empty
        return []
    
    # Build S(5,6,12) from S(4,5,11)
    all_11 = set(range(11))
    result = []
    for B in s4511:
        B_set = set(B)
        result.append(sorted(B_set | {11}))
        result.append(sorted(all_11 - B_set))
    
    return result
