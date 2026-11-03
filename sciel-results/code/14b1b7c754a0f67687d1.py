import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    points = list(range(v))
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    
    # Precomputation cost: O(n_blocks * C(k,t))
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # 3. Greedy Set Cover with Randomized Ties
    # We will run multiple greedy passes and keep the best one.
    best_selected = None
    best_count = float('inf')
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0: # Leave time for local search
            break
            
        uncovered_mask = (1 << n_t) - 1
        selected_block_indices = []
        
        while uncovered_mask:
            if time.time() > deadline - 30.0:
                break
                
            best_block_idx = -1
            best_gain = 0
            candidates = []
            
            # Iterate through all blocks
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_block_idx == -1:
                break
            
            # Randomly select from top candidates to diversify
            if len(candidates) > 1:
                selected_idx = random.choice(candidates)
            else:
                selected_idx = best_block_idx
                
            selected_block_indices.append(selected_idx)
            uncovered_mask &= ~block_masks[selected_idx]
            
        if uncovered_mask == 0:
            # Valid cover found
            if len(selected_block_indices) < best_count:
                best_count = len(selected_block_indices)
                best_selected = selected_block_indices.copy()
                
    if best_selected is None:
        # Fallback: If no valid cover found (shouldn't happen), use the last partial one or force complete
        # But for C(19,8,5) it should always find one.
        # If we ran out of time in the loop, we need to ensure validity.
        # The loop above breaks if time is low.
        # If uncovered_mask != 0, we have an invalid cover.
        # We must complete it.
        if 'uncovered_mask' in locals() and uncovered_mask:
             # Fast completion
             while uncovered_mask and time.time() < deadline - 5.0:
                 best_block_idx = -1
                 for i in range(n_blocks):
                     if (uncovered_mask & block_masks[i]).bit_count() > 0:
                         best_block_idx = i
                         break
                 if best_block_idx == -1:
                     break
                 selected_block_indices.append(best_block_idx)
                 uncovered_mask &= ~block_masks[best_block_idx]
             if uncovered_mask == 0:
                 best_selected = selected_block_indices
             else:
                 # Emergency: Return a trivial cover (all blocks) if needed, but that's huge.
                 # Better: Just return what we have if we are desperate, but it will be invalid.
                 # We should avoid this.
                 pass
                 
    if best_selected is None:
        return []
        
    selected_block_indices = best_selected
    
    # 4. Local Search: Remove redundant blocks
    # A block is redundant if every t-subset it covers is covered by at least one other selected block.
    
    # Initialize counts
    counts = [0] * n_t
    for b_idx in selected_block_indices:
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    # Iterative removal
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        new_selected = []
        for b_idx in selected_block_indices:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                # Remove this block
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
                
        selected_block_indices = new_selected
        
    # 5. Local Search: Swap Heuristic
    # Try to swap a block with another to enable further removals or just reduce count?
    # Swapping 1 for 1 doesn't reduce count directly, but might allow redundancy removal.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(selected_block_indices)):
            if time.time() > deadline - 2.0:
                break
            b_idx = selected_block_indices[i]
            mask = block_masks[b_idx]
            
            # Check if redundant
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                # Remove
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                selected_block_indices.pop(i)
                improved = True
                continue
            
            # Not redundant. Try to swap?
            # Calculate "lost" subsets if we remove b_idx
            lost_mask = 0
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    lost_mask |= (1 << idx)
                m ^= lsb
                
            if lost_mask:
                # Find best block to cover lost_mask
                best_swap_idx = -1
                best_swap_gain = 0
                # Sample some blocks to find a good swap
                # To save time, we only try if we have enough time
                if time.time() < deadline - 5.0:
                    sample_size = min(100, n_blocks)
                    for _ in range(sample_size):
                        cand_idx = random.randint(0, n_blocks - 1)
                        if cand_idx in selected_block_indices:
                            continue
                        gain = (lost_mask & block_masks[cand_idx]).bit_count()
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_idx = cand_idx
                    
                    if best_swap_idx != -1 and best_swap_gain > 0:
                        # Perform swap
                        # Remove b_idx
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                        selected_block_indices[i] = best_swap_idx
                        
                        # Add best_swap_idx
                        new_mask = block_masks[best_swap_idx]
                        m = new_mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lsb
                            
                        improved = True
                        # Break to restart loop as indices changed
                        break
                        
    # 6. Final Redundancy Removal
    # Re-initialize counts
    counts = [0] * n_t
    for b_idx in selected_block_indices:
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for b_idx in selected_block_indices:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
        selected_block_indices = new_selected
        
    result_blocks = [list(all_blocks[i]) for i in selected_block_indices]
    
    return result_blocks
