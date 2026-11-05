import itertools
import sys
import heapq
import random
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Precomputation
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    block_masks = [0] * n_k_blocks
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    def greedy_cover(seed=None):
        """
        Greedy Set Cover using a Lazy Max-Heap.
        If seed is provided, it introduces randomization in tie-breaking or initial order.
        """
        uncovered = (1 << n_t_subs) - 1
        selected_indices = []
        
        # Initialize heap with (-gain, block_index)
        # We want max gain, so we store negative gain for min-heap.
        # To add randomization, we can slightly perturb gains or shuffle initial order.
        # Simplest randomization: shuffle the block indices before computing initial gains?
        # No, gains depend on 'uncovered'.
        # Better: If gains are equal, pick randomly.
        
        heap = []
        if seed is not None:
            random.seed(seed)
        
        # Initial gains
        # We can speed this up by not pushing all if we use a different strategy, 
        # but 75k pushes is fine.
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > 0:
                # Add small random noise to break ties if seed is provided?
                # Or just rely on index order.
                # To make it a proper LSAP/Randomized heuristic, let's add tiny noise.
                if seed is not None:
                    noise = random.random() * 1e-9
                    gain_val = gain + noise
                else:
                    gain_val = gain
                heapq.heappush(heap, (-gain_val, i))
                
        while uncovered and heap:
            neg_gain, idx = heapq.heappop(heap)
            gain = -neg_gain
            
            # Verify if this gain is still valid
            # The stored gain is an upper bound. 
            # Actual gain = (block_masks[idx] & uncovered).bit_count()
            actual_gain = (block_masks[idx] & uncovered).bit_count()
            
            if actual_gain != int(gain): # Note: gain might have noise. 
                # If seed is None, gain is int. If seed is not None, gain is float.
                # We should store the integer gain for comparison or just recompute.
                # Let's store (negative_float_gain, idx, original_int_gain) to be safe?
                # Or simpler: just check if actual_gain == 0, or if actual_gain < stored_gain.
                # Since we only pop the max, if actual_gain < stored_gain, it's stale.
                if actual_gain < int(gain):
                    # Stale, re-push with updated gain
                    if actual_gain > 0:
                         if seed is not None:
                             noise = random.random() * 1e-9
                             new_gain_val = actual_gain + noise
                         else:
                             new_gain_val = actual_gain
                         heapq.heappush(heap, (-new_gain_val, idx))
                    continue
            
            if actual_gain == 0:
                continue
                
            # Select this block
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
            # We don't need to update all other blocks in the heap immediately (Lazy Heap).
            # They will be updated when popped.
            
        return selected_indices

    def local_reduce(indices):
        """
        Local Search: Remove redundant blocks.
        A block is redundant if all t-subsets it covers are covered by at least one other selected block.
        """
        if not indices:
            return []
            
        # Convert to set for O(1) lookup? No, we need to iterate.
        # We need to compute counts.
        counts = [0] * n_t_subs
        
        # Build counts
        for idx in indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
                
        selected_set = set(indices)
        # To allow efficient removal, we keep a list and a set.
        # But we are modifying the list.
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                m = block_masks[idx]
                essential = False
                # Check if any subset covered by this block has count == 1
                # If count == 1, this block is essential for that subset.
                mm = m
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] == 1:
                        essential = True
                        break
                    mm ^= lsb
                    
                if essential:
                    new_indices.append(idx)
                else:
                    # Redundant, remove it
                    # Decrement counts
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        mm ^= lsb
                    changed = True
            
            indices = new_indices
            
        return indices

    # Main Execution
    start_time = time.time()
    time_limit = 110.0 # Leave some buffer for overhead
    
    # 1. Deterministic Greedy
    best_solution = greedy_cover(seed=None)
    best_solution = local_reduce(best_solution)
    best_size = len(best_solution)
    
    # 2. Randomized Greedy + Local Search
    # Estimate time per run
    # Greedy is ~O(N log N) with lazy updates.
    # Local Reduce is ~O(K * C) where K is num blocks, C is bits per block.
    
    # Run several random instances
    num_runs = 20
    
    for i in range(num_runs):
        if time.time() - start_time > time_limit:
            break
            
        seed = i + 1
        sol = greedy_cover(seed=seed)
        sol = local_reduce(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
    return [list(k_blocks[i]) for i in best_solution]
