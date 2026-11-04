import itertools

def construct(v, k, t):
    # Generate all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    blocks = []
    
    while uncovered:
        # 1. Identify the most constrained uncovered subset.
        # Using min() is a simple heuristic.
        first_uncovered = min(uncovered)
        
        # 2. Generate all candidate blocks that contain first_uncovered.
        # A block B must contain first_uncovered.
        # So B = first_uncovered U R, where R is a (k-t) subset of the remaining v-t points.
        remaining_points = [p for p in range(v) if p not in first_uncovered]
        
        best_block = None
        max_gain = -1
        
        # Generate all (k-t) subsets of remaining_points
        for r_subset in itertools.combinations(remaining_points, k - t):
            # Construct the block
            block_set = set(first_uncovered)
            block_set.update(r_subset)
            # Convert to sorted tuple for consistency
            block = tuple(sorted(block_set))
            
            # Count how many uncovered t-subsets are in this block
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = block
                
        # 3. Add the best block
        blocks.append(list(best_block))
        
        # 4. Update uncovered set
        for sub in itertools.combinations(best_block, t):
            if sub in uncovered:
                uncovered.discard(sub)
                
    return blocks
