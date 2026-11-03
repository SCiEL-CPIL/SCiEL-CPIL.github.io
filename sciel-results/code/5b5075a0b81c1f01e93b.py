"""Covering design C(19,8,5): Optimized bitmask greedy + local search."""
import itertools
import random
import time

def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # Safety margin under 120s limit

    # 1. Precomputation
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    block_masks = [0] * num_blocks
    block_total_counts = [0] * num_blocks
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        block_total_counts[i] = mask.bit_count()
        
    # Sort block indices by total coverage (descending)
    # This helps in the greedy phase to focus on "good" blocks
    sorted_block_indices = sorted(range(num_blocks), key=lambda x: block_total_counts[x], reverse=True)
    
    # 2. Optimized Greedy Set Cover
    uncovered_mask = (1 << num_subsets) - 1
    selected_blocks = []
    counts = [0] * num_subsets
    
    # We will search in a subset of blocks to speed up the greedy step
    # Top K blocks by total count + Random samples
    K_TOP = 3000
    K_RAND = 3000
    
    while uncovered_mask:
        if time.time() > deadline - 20: # Reserve time for local search
            break
        
        best_block_idx = -1
        best_gain = 0
        
        # Candidate 1: Top K blocks
        for i in sorted_block_indices[:K_TOP]:
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        # Candidate 2: Random blocks (to escape potential biases)
        # Use a fixed seed for reproducibility in the random part if needed, 
        # but random is fine for diversity.
        for _ in range(K_RAND):
            i = random.randint(0, num_blocks - 1)
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1 or best_gain == 0:
            # Fallback: full search if we are stuck (should be rare)
            for i in range(num_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1 or best_gain == 0:
                break

        # Select the block
        selected_blocks.append(best_block_idx)
        mask = block_masks[best_block_idx]
        
        # Update counts
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
        # Update uncovered mask
        uncovered_mask &= ~mask

    # 3. Local Search: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
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
                
        selected_blocks = new_selected

    # 4. Local Search: Swap blocks to enable further removals
    # Strategy: For each block, identify its "unique" t-subsets (count == 1).
    # Try to find another block that covers all these unique t-subsets.
    # If found, swap the block. This keeps coverage valid.
    # Then, re-run the removal phase.
    
    if time.time() < deadline - 5:
        # Pre-compute list of unique t-subsets for each selected block
        # This is expensive, so we do it on the fly or cache it if we iterate multiple times.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            new_selected = list(selected_blocks)
            
            for i in range(len(new_selected)):
                if time.time() > deadline - 2:
                    break
                    
                b_idx = new_selected[i]
                mask = block_masks[b_idx]
                
                # Find unique t-subsets for this block
                unique_subs = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        unique_subs.append(idx)
                    m ^= lsb
                    
                if not unique_subs:
                    # Block is fully redundant? Should have been removed in phase 3.
                    # But if it wasn't, it might be due to order. Let's try to remove it.
                    # Actually, if unique_subs is empty, it's redundant.
                    # We can just remove it.
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    new_selected.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                    
                # We need to find a block B != A such that B covers all unique_subs
                # To speed this up, we can intersect masks.
                # Create a mask of unique subsets
                unique_mask = 0
                for idx in unique_subs:
                    unique_mask |= (1 << idx)
                    
                # Search for a replacement block
                # We can search in the top K blocks first
                found_swap = False
                # Check top blocks
                for j in sorted_block_indices[:1000]:
                    if j == b_idx:
                        continue
                    # Check if block j covers all unique subsets
                    if (block_masks[j] & unique_mask) == unique_mask:
                        # Valid swap
                        # Remove old block
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                            
                        # Add new block
                        new_mask = block_masks[j]
                        m = new_mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lsb
                            
                        new_selected[i] = j
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break # Restart loop to re-evaluate removals
                
        selected_blocks = new_selected
        # Final removal pass
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_selected = []
            for b_idx in selected_blocks:
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
            selected_blocks = new_selected

    result = [list(all_blocks[i]) for i in selected_blocks]
    return result
