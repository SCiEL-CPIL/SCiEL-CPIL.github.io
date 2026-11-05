import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Parameters for this specific task are v=19, k=7, t=4
    # But the function signature is generic. We assume the evaluator passes 19, 7, 4.
    
    # 1. Generate all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_count = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # 2. Generate all k-subsets (blocks) and compute their coverage masks
    # We use Python's arbitrary precision integers as bitmasks.
    # Bit i is set if the block covers the i-th t-subset.
    
    all_k_subs = list(itertools.combinations(range(v), k))
    k_sub_count = len(all_k_subs)
    
    # Precompute masks for all k-subsets
    # This might take a few seconds for large v, but for v=19 it's fast.
    block_masks = []
    for block in all_k_subs:
        mask = 0
        # Iterate over all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # 3. Greedy Set Cover
    # Initial state: all t-subsets are uncovered
    # We represent uncovered set as a bitmask where bit i is 1 if uncovered.
    # So initially, all bits are 1.
    full_mask = (1 << t_sub_count) - 1
    uncovered_mask = full_mask
    
    selected_blocks = []
    selected_indices = set()
    
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        
        # Find the block that covers the most uncovered t-subsets
        # We can skip blocks already selected, but their gain will be 0 anyway 
        # because their bits are cleared in uncovered_mask.
        # However, checking all blocks is O(N).
        
        # Optimization: If we have many blocks, we might want to maintain a list of candidates.
        # But for 50k blocks, a linear scan is fine.
        
        for i in range(k_sub_count):
            # Calculate gain: number of bits set in (uncovered_mask AND block_masks[i])
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early exit if we found a perfect block? 
                # Max possible gain is t_sub_count, but practically limited by block size.
                # A block covers at most C(k,t) t-subsets.
                # If gain == C(k,t), it's as good as it gets for this block.
                # But another block might have the same gain. We just pick the first.
        
        if best_idx == -1:
            # This shouldn't happen if uncovered_mask is not 0, unless v,k,t are inconsistent
            break
            
        # Add the block to the solution
        selected_blocks.append(list(all_k_subs[best_idx]))
        selected_indices.add(best_idx)
        
        # Update uncovered mask
        uncovered_mask &= ~block_masks[best_idx]
        
        # Time check
        # If we are running out of time, we can't return an invalid solution.
        # The greedy algorithm is deterministic and fast.
        # We only break if we hit a hard limit and MUST return something.
        # But returning a partial cover is invalid.
        # So we should only return if we are done.
        # Given the speed, we should always finish.
        if time.time() - start_time > 110:
            # If we are close to timeout, we assume we are stuck or slow.
            # But for v=19, it's fast.
            # If we really hit this, we might have to return the best valid cover we have?
            # No, a partial cover is invalid.
            # We must finish.
            pass

    return selected_blocks
