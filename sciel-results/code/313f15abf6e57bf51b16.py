import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, 5s margin for safety
    
    points = range(v)
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute for each t-subset, the list of points NOT in it? 
    # Or better: for a given t-subset S, the blocks containing S are formed by choosing k-t points from v-t points.
    
    def get_blocks_containing(subset):
        """Yields all k-subsets that contain the given t-subset."""
        subset_set = set(subset)
        remaining_points = [p for p in points if p not in subset_set]
        for combo in itertools.combinations(remaining_points, k - t):
            yield list(set(subset) | set(combo))

    best_blocks = None
    best_size = float('inf')
    
    # Multiple restarts of the greedy algorithm
    while time.time() < deadline:
        # uncovered[i] is 1 if t_subsets[i] is not yet covered
        uncovered = [1] * n_subsets
        blocks = []
        
        # Greedy construction
        while True:
            # Check if all covered
            # Optimization: maintain a count of uncovered? 
            # For now, checking sum(uncovered) > 0 is O(N). 
            # N=3876 is small.
            if not any(uncovered):
                break
            
            # Pick a random uncovered subset
            uncovered_indices = [i for i, val in enumerate(uncovered) if val == 1]
            if not uncovered_indices:
                break
            
            target_idx = random.choice(uncovered_indices)
            target_subset = t_subsets[target_idx]
            
            # Find best block containing target_subset
            best_gain = -1
            best_block = None
            candidates = list(get_blocks_containing(target_subset))
            
            for block in candidates:
                # Calculate gain: how many currently uncovered subsets does this block cover?
                gain = 0
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                # Should not happen, but safety break
                break
                
            blocks.append(best_block)
            
            # Update uncovered status
            for sub in itertools.combinations(best_block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = 0
        
        # If we didn't cover everything, skip this attempt
        if any(uncovered):
            continue
            
        # Pruning Phase
        # We want to remove redundant blocks.
        # A block is redundant if every t-subset in it is covered by at least one other block.
        
        # Count coverage for each t-subset
        counts = [0] * n_subsets
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                counts[idx] += 1
                
        # Identify redundant blocks
        new_blocks = []
        for block in blocks:
            is_redundant = True
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if counts[idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_blocks.append(block)
        
        # Second pass of pruning might be needed if removing a block makes another redundant
        if len(new_blocks) < len(blocks):
             blocks = new_blocks
             # Re-run pruning once more if time permits
             if time.time() < deadline - 2:
                 counts = [0] * n_subsets
                 for block in blocks:
                     for sub in itertools.combinations(block, t):
                         idx = subset_to_idx[sub]
                         counts[idx] += 1
                 
                 final_blocks = []
                 for block in blocks:
                     is_redundant = True
                     for sub in itertools.combinations(block, t):
                         idx = subset_to_idx[sub]
                         if counts[idx] == 1:
                             is_redundant = False
                             break
                     if not is_redundant:
                         final_blocks.append(block)
                 blocks = final_blocks

        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        # Fallback: Return a valid covering if we have one, otherwise empty (invalid)
        # But greedy should always find one quickly.
        return []
        
    # Ensure distinct blocks (greedy might add duplicates? Unlikely with randomization, but safe to check)
    # The problem states blocks must be distinct.
    seen = set()
    distinct_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            distinct_blocks.append(b)
            
    return distinct_blocks
