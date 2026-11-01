import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    block_points = all_k_blocks
    
    # block_bitmasks[i] is an integer bitmask where bit j is 1 if t_subsets[j] is in all_k_blocks[i]
    block_bitmasks = [0] * n_k_blocks
    
    # Optimization: Precompute the list of t-subset indices for each block
    # This is much faster than iterating over combinations and looking up dict each time in loops
    block_t_indices = [None] * n_k_blocks
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_bitmasks[i] = mask
        block_t_indices[i] = indices

    def greedy_cover(uncovered_mask, existing_blocks=None, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        if existing_blocks is None:
            existing_blocks = []
        
        selected_blocks = list(existing_blocks)
        
        while uncovered_mask:
            if time.time() > time_limit:
                break
                
            best_idx = -1
            best_gain = 0
            
            # Optimization: Only consider blocks not already in selected_blocks?
            # No, because a block in selected_blocks might have been added, but we are tracking
            # uncovered_mask which is updated. So any block in selected_blocks has 0 gain.
            # We can skip them to save time?
            # Creating a set of selected indices is O(N). Checking membership is O(1).
            # But N is growing.
            # Let's just compute gain. If gain is 0, it won't be selected.
            
            for i in range(n_k_blocks):
                # Quick check: if block is already selected, skip?
                # We don't have a fast "is selected" check without a set.
                # Let's just compute gain.
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break # Should not happen if uncovered is not empty and blocks cover universe
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_bitmasks[best_idx]
            
        return selected_blocks, uncovered_mask

    def local_search_remove(blocks_indices, time_limit=None):
        if time_limit is None:
            time_limit = deadline
            
        # counts[j] = number of selected blocks covering t-subset j
        # Use a dictionary or array? n_t_subsets is C(19,5) = 11628.
        # Array is better.
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        
        for b_idx in blocks_indices:
            for idx in block_t_indices[b_idx]:
                counts[idx] += 1
                
        # Iterate until no more removals
        changed = True
        while changed:
            if time.time() > time_limit:
                break
            changed = False
            to_remove = []
            
            for b_idx in blocks_indices:
                # Check if any t-subset in this block has count == 1
                # If so, it's essential
                essential = False
                for idx in block_t_indices[b_idx]:
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                # Remove them
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                changed = True
                
        return blocks_indices, counts

    def swap_search(blocks_indices, counts, time_limit=None):
        # Try to replace one block with another to reduce total count?
        # Or replace one block with a different block that covers the same unique parts?
        # This is complex. Let's stick to removal first.
        # Maybe try adding a block and removing two?
        # That's a 1-2 swap.
        # If we add a block B, it covers some subsets.
        # If those subsets were only covered by B1 and B2 (count 2), and B covers them,
        # we might be able to remove B1 and B2 if they become redundant?
        # This is expensive.
        
        # Let's try a simpler improvement: 
        # If we have a block that is "almost" redundant (all its subsets are covered by 2+ blocks),
        # but removing it causes some subsets to drop to count 1, maybe we can swap it with a block
        # that covers those "critical" subsets?
        
        # For now, let's focus on making removal faster and doing more restarts.
        
        return blocks_indices

    # Run Greedy
    initial_blocks, _ = greedy_cover((1 << n_t_subsets) - 1)
    
    # Run Local Search
    current_blocks, counts = local_search_remove(initial_blocks)
    
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    # If time remains, try randomized restarts or perturbations
    # Strategy: Randomly remove a few blocks, then re-greedy the gaps.
    
    def cover_holes(existing_blocks, holes_mask, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        new_blocks = []
        uncovered = holes_mask
        while uncovered:
            if time.time() > time_limit:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                gain = (uncovered & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            uncovered &= ~block_bitmasks[best_idx]
        return new_blocks

    # Perform perturbation attempts
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Create a copy of current best
        perturbed_blocks = list(best_blocks)
        
        # Randomly remove a few blocks
        # How many? Maybe 2-5?
        # If we remove too many, we add too many back.
        # If we remove too few, we might not escape local minima.
        # Let's try removing 1-3 blocks.
        num_remove = min(len(perturbed_blocks), random.randint(1, 3))
        if num_remove == 0:
            break
        to_remove = random.sample(perturbed_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in perturbed_blocks if b not in remove_set]
        
        # Calculate holes
        covered_mask = 0
        for b_idx in perturbed_blocks:
            covered_mask |= block_bitmasks[b_idx]
            
        holes_mask = ((1 << n_t_subsets) - 1) & ~covered_mask
        
        # Cover holes
        added_blocks = cover_holes(perturbed_blocks, holes_mask)
        
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)
        
        if new_count < best_count:
            # Run local search on the new set to clean it up
            cleaned_blocks, _ = local_search_remove(new_total_blocks)
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_count = len(best_blocks)
            else:
                # If local search didn't improve, check if raw new_total is better
                if len(new_total_blocks) < best_count:
                    best_blocks = new_total_blocks
                    best_count = len(best_blocks)

    # Convert indices back to point lists
    final_blocks = [list(block_points[i]) for i in best_blocks]
    
    return final_blocks
