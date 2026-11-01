import itertools
import time
import random
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks and their bitmasks of covered t-subsets
    # Also store the actual points of the block for output
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_bitmasks[i] is an integer bitmask where bit j is 1 if t_subsets[j] is in all_k_blocks[i]
    # block_points[i] is the tuple of points
    block_bitmasks = [0] * n_k_blocks
    block_points = all_k_blocks
    
    # Map points to t-subsets containing them? Not needed for bitmask greedy.
    # We need to build the bitmasks.
    # For each block, find all t-subsets within it.
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks[i] = mask
    
    # Build conflict graph: two blocks conflict if they share any t-subset
    # For MWIS, we want to find a set of non-conflicting blocks that covers everything
    # Actually, for covering design, we need ALL t-subsets covered, so we can't just do MWIS
    # Instead, let's use a different approach: branch-and-bound for minimum set cover with symmetry breaking
    
    # For symmetry breaking: fix that block 0 contains point 0, block 1 contains points 0,1, etc.
    # This is complex, so let's use a simpler approach: greedy + local search with better initialization

    def greedy_cover():
        uncovered_mask = (1 << n_t_subsets) - 1
        selected_blocks = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_gain = 0
            
            # Iterate through all blocks to find the one covering most uncovered subsets
            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break # Should not happen if uncovered is not empty and blocks cover universe
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_bitmasks[best_idx]
            
        return selected_blocks
    
    def greedy_cover_with_bias(biased_blocks=None):
        """Greedy cover with preference for blocks in biased_blocks"""
        uncovered_mask = (1 << n_t_subsets) - 1
        selected_blocks = []
        biased_set = set(biased_blocks) if biased_blocks else set()
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_gain = -1
            
            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                # Prefer biased blocks with a small bonus
                if i in biased_set:
                    gain += 0.1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_bitmasks[best_idx]
            
        return selected_blocks

    def local_search_remove(blocks_indices):
        # counts[j] = number of selected blocks covering t-subset j
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        
        for b_idx in blocks_indices:
            mask = block_bitmasks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
        # Iterate until no more removals
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            
            for b_idx in blocks_indices:
                mask = block_bitmasks[b_idx]
                m = mask
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                # Remove them
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    mask = block_bitmasks[b_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                changed = True
                
        return blocks_indices
    
    def local_search_swap(blocks_indices):
        """Try swapping one block for another to reduce total count"""
        if time.time() > deadline:
            return blocks_indices
        
        # Compute coverage counts
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            mask = block_bitmasks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        selected_set = set(blocks_indices)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i, b_idx in enumerate(list(blocks_indices)):
                # Try removing b_idx
                mask_remove = block_bitmasks[b_idx]
                m = mask_remove
                uncovered_after_remove = []
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        uncovered_after_remove.append(idx)
                    m ^= lsb
                
                if not uncovered_after_remove:
                    # Can remove without adding
                    blocks_indices = blocks_indices[:i] + blocks_indices[i+1:]
                    m = mask_remove
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    selected_set.discard(b_idx)
                    improved = True
                    break
                
                # Try to find a block that covers all uncovered_after_remove
                # and doesn't add too much new coverage
                target_mask = 0
                for idx in uncovered_after_remove:
                    target_mask |= (1 << idx)
                
                best_swap = -1
                best_new_gain = float('inf')
                
                for j in range(n_k_blocks):
                    if j in selected_set:
                        continue
                    mask_j = block_bitmasks[j]
                    # Check if mask_j covers all uncovered_after_remove
                    if (mask_j & target_mask) != target_mask:
                        continue
                    # Count how many new t-subsets this block covers (not already covered)
                    new_cover = (mask_j & ~block_bitmasks[b_idx]) & ((1 << n_t_subsets) - 1)
                    # Actually, we need to check what's covered by other blocks
                    # This is complex, so let's just try the swap and see
                    best_swap = j
                    break
                
                if best_swap != -1:
                    # Do the swap
                    blocks_indices[i] = best_swap
                    selected_set.add(best_swap)
                    selected_set.discard(b_idx)
                    
                    # Update counts
                    m = block_bitmasks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    
                    m = block_bitmasks[best_swap]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    
                    improved = True
                    break
        
        return blocks_indices

    # Run Greedy
    initial_blocks = greedy_cover()
    
    # Run Local Search
    current_blocks = local_search_remove(initial_blocks)
    
    # Try swap-based local search
    current_blocks = local_search_swap(current_blocks)
    
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    def cover_holes(existing_blocks, holes_mask):
        # holes_mask is a bitmask of t-subsets that are currently UNCOVERED
        new_blocks = []
        uncovered = holes_mask
        while uncovered:
            if time.time() > deadline:
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
    while time.time() < deadline - 5:
        attempts += 1
        perturbed_blocks = list(best_blocks)
        
        # Randomly remove 3-8 blocks
        num_remove = min(len(perturbed_blocks), random.randint(3, 8))
        if num_remove == 0:
            break
        to_remove = random.sample(perturbed_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in perturbed_blocks if b not in remove_set]
        
        # Compute holes
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
            cleaned_blocks = local_search_remove(new_total_blocks)
            cleaned_blocks = local_search_swap(cleaned_blocks)
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_count = len(best_blocks)
            else:
                if len(new_total_blocks) < best_count:
                    best_blocks = new_total_blocks
                    best_count = len(best_blocks)
        
        # Also try biased greedy with some of the current blocks
        if time.time() < deadline - 10 and len(best_blocks) > 10:
            # Use a subset of current blocks as bias
            bias_size = max(5, len(best_blocks) // 4)
            biased = random.sample(best_blocks, bias_size)
            biased_blocks = greedy_cover_with_bias(biased)
            biased_blocks = local_search_remove(biased_blocks)
            biased_blocks = local_search_swap(biased_blocks)
            if len(biased_blocks) < best_count:
                best_blocks = biased_blocks
                best_count = len(best_blocks)
    
    # Convert indices back to point lists
    final_blocks = [list(block_points[i]) for i in best_blocks]
    
    return final_blocks
