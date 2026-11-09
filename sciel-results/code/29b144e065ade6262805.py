import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using an optimized greedy set cover
    with incremental gain updates, redundancy removal, and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(block_masks)
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    # 3. Optimized greedy with incremental gain tracking
    # For each t-subset, track how many chosen blocks cover it
    coverage_count = [0] * num_t_subsets
    
    # For each block, precompute which t-subsets it covers (as a list of indices)
    block_t_indices = []
    for i in range(num_blocks):
        mask = block_masks[i]
        indices = []
        for j in range(num_t_subsets):
            if mask & (1 << j):
                indices.append(j)
        block_t_indices.append(indices)
    
    def greedy_with_tiebreak(tie_mode):
        """Greedy set cover with different tie-breaking strategies."""
        nonlocal coverage_count
        coverage_count = [0] * num_t_subsets
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_indices = []
        chosen_masks = []
        
        while current_uncovered != 0:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_block_idx = -1
            best_tie = None
            
            for i in range(num_blocks):
                # Compute gain: number of currently uncovered t-subsets this block covers
                gain = 0
                for j in block_t_indices[i]:
                    if coverage_count[j] == 0:
                        gain += 1
                
                if gain <= 0:
                    continue
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    if tie_mode == 'overlap':
                        best_tie = count_bits(block_masks[i] & ~current_uncovered)
                    elif tie_mode == 'diversity':
                        best_tie = -count_bits(block_masks[i] & ~current_uncovered)
                    else:
                        best_tie = i
                elif gain == best_gain:
                    if tie_mode == 'overlap':
                        tie = count_bits(block_masks[i] & ~current_uncovered)
                    elif tie_mode == 'diversity':
                        tie = -count_bits(block_masks[i] & ~current_uncovered)
                    else:
                        tie = i
                    
                    if best_tie is None or tie > best_tie:
                        best_tie = tie
                        best_block_idx = i
            
            if best_block_idx == -1:
                break
            
            chosen_indices.append(best_block_idx)
            chosen_masks.append(block_masks[best_block_idx])
            
            # Update coverage counts
            for j in block_t_indices[best_block_idx]:
                coverage_count[j] += 1
            
            current_uncovered &= ~block_masks[best_block_idx]
        
        return chosen_indices, chosen_masks
    
    def remove_redundant(chosen_indices, chosen_masks):
        """Remove redundant blocks from the solution."""
        n = len(chosen_indices)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if time.time() > deadline - 3.0:
                    break
                
                mask_i = chosen_masks[i]
                
                # Calculate union of others
                union_others = 0
                for j in range(n):
                    if i != j:
                        union_others |= chosen_masks[j]
                
                # Check if mask_i is subset of union_others
                if (mask_i & ~union_others) == 0:
                    chosen_indices.pop(i)
                    chosen_masks.pop(i)
                    n -= 1
                    changed = True
                    break
        
        return chosen_indices, chosen_masks
    
    def local_search_2for1(chosen_indices, chosen_masks):
        """Try to replace pairs of blocks with single blocks to reduce count."""
        n = len(chosen_indices)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Combined coverage of blocks i and j
                    combined = chosen_masks[i] | chosen_masks[j]
                    
                    # Check if any single block covers all of 'combined'
                    # that is not already in the chosen set
                    for b in range(num_blocks):
                        if b in chosen_indices:
                            continue
                        if (block_masks[b] | combined) == combined:
                            # Block b covers everything i and j cover
                            # Remove i and j, add b
                            # But we need to be careful: b might be the same as i or j
                            new_indices = []
                            new_masks = []
                            for idx, m in zip(chosen_indices, chosen_masks):
                                if idx != chosen_indices[i] and idx != chosen_indices[j]:
                                    new_indices.append(idx)
                                    new_masks.append(m)
                            new_indices.append(b)
                            new_masks.append(block_masks[b])
                            
                            # Verify coverage is maintained
                            all_covered = 0
                            for m in new_masks:
                                all_covered |= m
                            if all_covered == (1 << num_t_subsets) - 1:
                                chosen_indices = new_indices
                                chosen_masks = new_masks
                                n = len(chosen_indices)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return chosen_indices, chosen_masks
    
    # Run multiple greedy variants and keep the best
    best_indices = None
    best_masks = None
    best_count = float('inf')
    
    # Try different tie-breaking modes
    for tie_mode in ['overlap', 'diversity', 'none']:
        if time.time() > deadline - 10.0:
            break
        
        indices, masks = greedy_with_tiebreak(tie_mode)
        indices, masks = remove_redundant(indices, masks)
        
        if len(indices) < best_count:
            best_count = len(indices)
            best_indices = indices
            best_masks = masks
    
    # Local search on the best solution
    if best_indices is not None and time.time() < deadline - 5.0:
        best_indices, best_masks = local_search_2for1(best_indices, best_masks)
        best_count = len(best_indices)
    
    # Final redundancy removal
    if best_indices is not None:
        best_indices, best_masks = remove_redundant(best_indices, best_masks)
    
    if best_indices is None:
        # Fallback: simple greedy
        indices, masks = greedy_with_tiebreak('overlap')
        indices, masks = remove_redundant(indices, masks)
        best_indices = indices
    
    return [list(all_blocks[idx]) for idx in best_indices]
