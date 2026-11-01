import itertools
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a greedy set cover algorithm
    optimized with bitmasks and candidate filtering.
    """
    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute all k-blocks, their masks, and reverse index (which blocks cover which t-subset)
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    # blocks_containing[i] holds indices of blocks that cover t_subset i
    blocks_containing = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        mask = 0
        # Iterate over all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            blocks_containing[idx].append(b_idx)
        block_masks.append(mask)
        
    # 3. Greedy Set Cover
    # uncovered: bitmask of t-subsets not yet covered
    uncovered = (1 << num_t_subsets) - 1
    blocks_used = []
    
    # Safety deadline: 110 seconds
    start_time = time.time()
    
    while uncovered:
        if time.time() - start_time > 110:
            break
            
        # Find an uncovered t-subset to anchor our candidate search
        # We pick the one with the lowest index that is uncovered
        # This is deterministic and fast
        first_bit = (uncovered & -uncovered).bit_length() - 1
        
        # Get candidates: blocks that contain this specific t-subset
        candidates = blocks_containing[first_bit]
        
        best_block_idx = -1
        best_gain = -1
        
        # Evaluate candidates
        for b_idx in candidates:
            mask = block_masks[b_idx]
            overlap = mask & uncovered
            if overlap:
                gain = overlap.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    # Early termination if we cover all remaining
                    if gain == uncovered.bit_count():
                        break
                        
        if best_block_idx == -1:
            # No block covers the first_bit? Should not happen if covering is possible
            break
            
        blocks_used.append(all_k_blocks[best_block_idx])
        uncovered &= ~block_masks[best_block_idx]
        
    return [list(b) for b in blocks_used]
