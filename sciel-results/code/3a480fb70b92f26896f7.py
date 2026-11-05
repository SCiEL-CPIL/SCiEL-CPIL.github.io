import itertools
import sys
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    block_masks = [0] * n_k_blocks
    block_mask_sizes = [0] * n_k_blocks # Popcount of mask
    
    # Precompute masks
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        block_mask_sizes[i] = mask.bit_count()
        
    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_block_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search: Iterative Redundancy Removal
    # Strategy: Repeatedly find a redundant block (one that covers no unique t-subset).
    # Remove the one with the smallest mask size to minimize impact on other blocks' redundancy.
    
    def compute_counts(indices):
        counts = [0] * n_t_subs
        for idx in indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
        return counts

    def is_redundant(idx, counts):
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            m ^= lsb
        return True

    # Iterative removal
    changed = True
    while changed:
        changed = False
        counts = compute_counts(selected_block_indices)
        
        # Find all redundant blocks
        redundant_blocks = []
        for idx in selected_block_indices:
            if is_redundant(idx, counts):
                redundant_blocks.append(idx)
                
        if redundant_blocks:
            # Sort by mask size (ascending) to remove "least impactful" first
            redundant_blocks.sort(key=lambda x: block_mask_sizes[x])
            # Remove the first one
            rem_idx = redundant_blocks[0]
            selected_block_indices.remove(rem_idx)
            changed = True
            # Optional: Break to recompute counts fully? 
            # Since removing one block only affects counts of its subsets,
            # we could update counts incrementally, but full recompute is O(N * 56) which is fast.
            
    # 5. Swap-Remove Search (Escape Local Minima)
    # If no redundant blocks, try to find a swap that creates one.
    # Time limit check
    start_time = time.time()
    timeout = 110.0 # Leave 10s for overhead
    
    # Precompute for each t-subset, which blocks cover it?
    # This is useful for finding candidates.
    # subset_to_blocks = [[] for _ in range(n_t_subs)]
    # for i, m in enumerate(block_masks):
    #     while m:
    #         lsb = m & -m
    #         bit_idx = lsb.bit_length() - 1
    #         subset_to_blocks[bit_idx].append(i)
    #         m ^= lsb
    # This takes 75582 * 56 ~ 4M ops. Fast.
    
    subset_to_blocks = [[] for _ in range(n_t_subs)]
    for i, m in enumerate(block_masks):
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            subset_to_blocks[bit_idx].append(i)
            m ^= lsb

    # Local Search Loop
    while time.time() - start_time < timeout:
        counts = compute_counts(selected_block_indices)
        
        # Check for redundancy again (in case swap created one)
        redundant_found = False
        for idx in selected_block_indices:
            if is_redundant(idx, counts):
                selected_block_indices.remove(idx)
                redundant_found = True
                break
        if redundant_found:
            continue # Restart loop to check for more
            
        # No redundant blocks. Try Swap.
        swap_performed = False
        # Sort selected blocks by number of unique subsets (smallest first) to try easy swaps
        # Unique subsets for B are those with count 1.
        unique_counts = {}
        for idx in selected_block_indices:
            m = block_masks[idx]
            cnt = 0
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    cnt += 1
                m ^= lsb
            unique_counts[idx] = cnt
            
        # Try blocks with fewest unique subsets (easiest to replace)
        candidates = sorted(selected_block_indices, key=lambda x: unique_counts[x])
        
        for b_out in candidates:
            if time.time() - start_time > timeout:
                break
                
            # Find unique subsets of b_out
            m_out = block_masks[b_out]
            unique_subs = []
            while m_out:
                lsb = m_out & -m_out
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_subs.append(bit_idx)
                m_out ^= lsb
                
            if not unique_subs:
                continue # Should be redundant, handled above
                
            # Find a block b_in that covers all unique_subs
            # Intersect the lists of blocks covering each unique sub
            # Start with the list of the rarest unique sub
            # Sort unique_subs by len(subset_to_blocks[s])
            unique_subs.sort(key=lambda s: len(subset_to_blocks[s]))
            
            # Candidates for b_in are in the intersection of subset_to_blocks[s] for s in unique_subs
            # And b_in must not be in selected_block_indices
            sel_set = set(selected_block_indices)
            
            # Get candidates from the first (smallest) list
            potential_ins = subset_to_blocks[unique_subs[0]]
            
            valid_in = -1
            for b_in in potential_ins:
                if b_in in sel_set:
                    continue
                # Check if b_in covers all other unique_subs
                m_in = block_masks[b_in]
                covers_all = True
                for s in unique_subs[1:]:
                    if not (m_in & (1 << s)):
                        covers_all = False
                        break
                if covers_all:
                    valid_in = b_in
                    break
                    
            if valid_in != -1:
                # Perform Swap: Remove b_out, Add b_in
                selected_block_indices.remove(b_out)
                selected_block_indices.append(valid_in)
                swap_performed = True
                break # Restart loop to check for redundancy
                
        if not swap_performed:
            # No swap found, break
            break
            
    # Final cleanup
    counts = compute_counts(selected_block_indices)
    redundant_found = True
    while redundant_found:
        redundant_found = False
        for idx in list(selected_block_indices):
            if is_redundant(idx, counts):
                selected_block_indices.remove(idx)
                # Update counts
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    m ^= lsb
                redundant_found = True

    return [list(k_blocks[i]) for i in selected_block_indices]
