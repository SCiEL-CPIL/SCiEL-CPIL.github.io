import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Precompute bitmasks for all k-subsets
    # A k-subset covers C(k, t) t-subsets
    # We represent the set of covered t-subsets as a Python integer bitmask
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute the bitmask for each k-subset
    # mask[i] is an integer where bit j is 1 if t_subsets[j] is a subset of k_subsets[i]
    masks = [0] * n_k
    
    # Optimization: Iterate over k-subsets and their t-subsets
    # For v=19, k=8, t=5:
    # n_k = 75582
    # Each k-subset has C(8,5)=56 t-subsets.
    # Total operations: 75582 * 56 ~ 4.2M dictionary lookups and bit sets. This is fast.
    
    for i, ksub in enumerate(k_subsets):
        mask_val = 0
        # Generate all t-subsets of this k-subset
        for tsub in itertools.combinations(ksub, t):
            idx = t_to_idx[tsub]
            mask_val |= (1 << idx)
        masks[i] = mask_val

    # 3. Greedy Construction
    # uncovered_mask starts with all bits set (all t-subsets uncovered)
    uncovered_mask = (1 << n_t) - 1
    blocks = []
    used_blocks = set()
    
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        # Find the block that covers the most currently uncovered t-subsets
        # We iterate over all blocks. To avoid re-picking the same block, we skip used ones.
        # Since we remove coverage, a previously used block will have 0 gain on uncovered parts 
        # (unless it covers something else? No, if it was picked, all its t-subsets were covered 
        # at that time. But other blocks might have covered some of its t-subsets before it was picked? 
        # No, greedy picks the best. Once picked, its contribution to 'uncovered' is removed.
        # So gain of a used block is always 0 on the current uncovered_mask.
        # We can just iterate all blocks and pick max gain. If gain is 0, we are stuck (shouldn't happen if uncovered_mask is not 0).
        
        # Optimization: Only check blocks not in used_blocks? 
        # Actually, if a block was used, `uncovered_mask & masks[i]` will be 0 because 
        # when we added block i, we did `uncovered_mask &= ~masks[i]`. 
        # So `uncovered_mask` has 0s where `masks[i]` had 1s.
        # Thus `uncovered_mask & masks[i]` is 0 for any used block i.
        # So we can just iterate all blocks.
        
        for i in range(n_k):
            gain = (uncovered_mask & masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_gain == 0:
            # Should not happen if uncovered_mask is not 0
            break
            
        # Add the block
        blocks.append(k_subsets[best_block_idx])
        used_blocks.add(best_block_idx)
        
        # Update uncovered mask
        uncovered_mask &= ~masks[best_block_idx]

    # 4. Redundancy Removal
    # If we have time, try to remove redundant blocks
    if time.time() < deadline - 5.0:
        # Count how many times each t-subset is covered
        # cover_counts[j] is the number of blocks in `blocks` that cover t_subsets[j]
        cover_counts = [0] * n_t
        
        # Precompute list of t-subset indices for each block in the final list
        block_t_indices = []
        for b in blocks:
            idxs = []
            for tsub in itertools.combinations(b, t):
                idxs.append(t_to_idx[tsub])
            block_t_indices.append(idxs)
            
            for idx in idxs:
                cover_counts[idx] += 1
        
        # Iteratively remove redundant blocks
        removed = True
        while removed:
            removed = False
            if time.time() > deadline - 2.0:
                break
                
            # Iterate through blocks backwards to allow removal
            for i in range(len(blocks) - 1, -1, -1):
                is_redundant = True
                for idx in block_t_indices[i]:
                    if cover_counts[idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    for idx in block_t_indices[i]:
                        cover_counts[idx] -= 1
                    
                    # Remove from lists
                    blocks.pop(i)
                    block_t_indices.pop(i)
                    removed = True
                    break # Restart scan from end

    return blocks
