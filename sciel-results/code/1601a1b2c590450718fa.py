import itertools
import time
import random

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin

    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    # Precompute for each t-subset, which blocks cover it (for swap search)
    blocks_containing_subset = [[] for _ in range(n_t_subs)]

    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            blocks_containing_subset[idx].append(i)
        block_masks[i] = mask

    # Precompute gains for greedy: for each block, how many t-subsets it covers
    block_sizes = [mask.bit_count() for mask in block_masks]
        
    # 3. Greedy Construction with randomization to escape local optima
    best_solution = None
    best_size = float('inf')
    
    # Try multiple random orderings or tie-breaking strategies
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 50:
            break
            
        uncovered_mask = (1 << n_t_subs) - 1
        selected_block_indices = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 40:
                break
                
            best_idx = -1
            best_gain = -1
            
            # Add slight randomization to tie-breaking
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                
                # Small random perturbation for ties
                if gain == best_gain and best_idx != -1 and random.random() < 0.1:
                    # 10% chance to switch on tie
                    if random.random() < 0.5:
                        best_idx = i
                elif gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
                
            selected_block_indices.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        if len(selected_block_indices) < best_size:
            best_size = len(selected_block_indices)
            best_solution = selected_block_indices[:]
            
    if best_solution is None:
        # Fallback to standard greedy if no solution found
        uncovered_mask = (1 << n_t_subs) - 1
        selected_block_indices = []
        while uncovered_mask != 0:
            if time.time() > deadline - 20:
                break
            best_idx = -1
            best_gain = -1
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected_block_indices.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        best_solution = selected_block_indices
    
    # 4. Local Search: Redundancy Removal
    current_blocks = best_solution[:]
    current_masks = [block_masks[i] for i in current_blocks]
    
    cover_count = [0] * n_t_subs
    for mask in current_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_current_blocks = []
        new_current_masks = []
        
        for i in range(len(current_blocks)):
            mask = current_masks[i]
            is_critical = False
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_critical = True
                    break
                m ^= lsb
                
            if is_critical:
                new_current_blocks.append(current_blocks[i])
                new_current_masks.append(mask)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
                
        current_blocks = new_current_blocks
        current_masks = new_current_masks
        
    # 5. Swap-based local search: try replacing one block with another
    # This can sometimes reduce the count further by finding a block that covers
    # more unique subsets than the one being removed.
    if time.time() < deadline - 15:
        # Build a set of selected block indices for quick lookup
        selected_set = set(current_blocks)
        
        # For each selected block, find its critical subsets (those covered only by this block)
        critical_subsets_for_block = {}
        for i, block_idx in enumerate(current_blocks):
            mask = current_masks[i]
            crit = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    crit.append(idx)
                m ^= lsb
            critical_subsets_for_block[block_idx] = crit
        
        # Try single-block swaps: for each selected block A with critical subsets,
        # find an unselected block B that covers all of A's critical subsets.
        # If found, swap A with B, then re-run redundancy removal.
        # This might allow further removals.
        
        swapped = True
        while swapped and time.time() < deadline - 10:
            swapped = False
            
            # Rebuild cover_count and critical sets after each swap
            cover_count = [0] * n_t_subs
            for mask in current_masks:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            critical_subsets_for_block = {}
            for i, block_idx in enumerate(current_blocks):
                mask = current_masks[i]
                crit = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        crit.append(idx)
                    m ^= lsb
                critical_subsets_for_block[block_idx] = crit
            
            # Try swapping each selected block
            for i in range(len(current_blocks)):
                if time.time() > deadline - 10:
                    break
                    
                block_a = current_blocks[i]
                crit_a = critical_subsets_for_block[block_a]
                
                # If no critical subsets, this block is already removable (handled by step 4)
                if not crit_a:
                    continue
                
                # Find an unselected block B that covers all critical subsets of A
                # and provides additional coverage (to potentially enable more removals)
                best_b = -1
                best_extra_gain = 0
                
                crit_mask = 0
                for idx in crit_a:
                    crit_mask |= (1 << idx)
                
                for b_idx in range(n_k_blocks):
                    if b_idx in selected_set:
                        continue
                    
                    mask_b = block_masks[b_idx]
                    
                    # B must cover all critical subsets of A
                    if (mask_b & crit_mask) != crit_mask:
                        continue
                    
                    # Count how many additional subsets B covers (beyond what A covers)
                    # This could enable removal of other blocks
                    extra = (mask_b & ~current_masks[i]).bit_count()
                    
                    if extra > best_extra_gain:
                        best_extra_gain = extra
                        best_b = b_idx
                
                if best_b != -1 and best_extra_gain > 0:
                    # Perform the swap
                    selected_set.discard(block_a)
                    selected_set.add(best_b)
                    current_blocks[i] = best_b
                    current_masks[i] = block_masks[best_b]
                    swapped = True
                    break  # Restart the while loop to re-evaluate
            
            # After swaps, run redundancy removal again
            if swapped:
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    new_current_blocks = []
                    new_current_masks = []
                    
                    for i in range(len(current_blocks)):
                        mask = current_masks[i]
                        is_critical = False
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            if cover_count[idx] == 1:
                                is_critical = True
                                break
                            m ^= lsb
                            
                        if is_critical:
                            new_current_blocks.append(current_blocks[i])
                            new_current_masks.append(mask)
                        else:
                            m = mask
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_count[idx] -= 1
                                m ^= lsb
                            improved = True
                            
                    current_blocks = new_current_blocks
                    current_masks = new_current_masks
    
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
