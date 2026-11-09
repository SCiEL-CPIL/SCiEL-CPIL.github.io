import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    # Generate all t-subsets and map to masks
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_mask = {}
    mask_to_t_subset = {}
    for sub in all_t_subsets:
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_subset_to_mask[frozenset(sub)] = mask
        mask_to_t_subset[mask] = sub
        
    initial_uncovered = set(t_subset_to_mask.values())
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_subset_to_index = {frozenset(block): i for i, block in enumerate(all_k_subsets)}
    
    # Precompute covers for each k-subset
    block_covers = []
    for block in all_k_subsets:
        covered_masks = set()
        for t_sub in itertools.combinations(block, t):
            mask = 0
            for p in t_sub:
                mask |= (1 << p)
            covered_masks.add(mask)
        block_covers.append(covered_masks)
        
    # Precompute for each t-subset (mask), the list of k-block indices that contain it
    # This allows us to only check relevant blocks in the greedy step
    t_mask_to_block_indices = {mask: [] for mask in initial_uncovered}
    
    for i, block in enumerate(all_k_subsets):
        # For each t-subset in this block, add block index i to its list
        for t_sub in itertools.combinations(block, t):
            mask = 0
            for p in t_sub:
                mask |= (1 << p)
            t_mask_to_block_indices[mask].append(i)
            
    # 2. Greedy with restarts
    best_blocks = None
    best_count = float('inf')
    
    # Determine number of restarts based on time
    # Each step: ~455 candidates * intersection cost.
    # ~150 steps.
    # One run should be very fast (< 0.1s).
    # We can run many.
    
    while time.time() < deadline:
        uncovered = set(initial_uncovered)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to focus on
            # Using min() is deterministic, random.choice is better for diversification
            # But random.choice on a set requires converting to list or using itertools.islice
            # Since uncovered shrinks, and we do this 150 times, overhead is small.
            # Let's pick the first one (min) for speed, or random.
            # Random is better for escaping local optima.
            # To avoid O(N) for random choice, we can just pick min(uncovered) if we want speed, 
            # but min is also O(N). 
            # Actually, iterating to find min is fast in C.
            # Let's use min for speed, and rely on the block selection randomization.
            target_mask = min(uncovered)
            
            # Get candidate blocks that cover this target_mask
            candidates = t_mask_to_block_indices[target_mask]
            
            if not candidates:
                # Error: no block covers this? Should not happen.
                break
                
            # Calculate gain for each candidate
            max_gain = 0
            best_candidates = []
            
            for idx in candidates:
                cover = block_covers[idx]
                gain = len(uncovered & cover)
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [idx]
                elif gain == max_gain:
                    best_candidates.append(idx)
                    
            if not best_candidates:
                break
                
            # Randomly select from best candidates
            chosen_idx = random.choice(best_candidates)
            blocks.append(all_k_subsets[chosen_idx])
            uncovered.difference_update(block_covers[chosen_idx])
            
        if uncovered:
            continue
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    if best_blocks is None:
        # Fallback
        uncovered = set(initial_uncovered)
        blocks = []
        while uncovered:
            target_mask = min(uncovered)
            candidates = t_mask_to_block_indices[target_mask]
            max_gain = 0
            best_idx = candidates[0]
            for idx in candidates:
                gain = len(uncovered & block_covers[idx])
                if gain > max_gain:
                    max_gain = gain
                    best_idx = idx
            blocks.append(all_k_subsets[best_idx])
            uncovered.difference_update(block_covers[best_idx])
        return [list(b) for b in blocks]
        
    return best_blocks
