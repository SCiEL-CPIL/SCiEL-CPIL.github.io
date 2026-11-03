import itertools

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to bit indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    t_sub_to_bit = {ts: i for i, ts in enumerate(t_subs)}
    
    # 2. Generate all k-subsets (blocks) and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    block_masks = []
    for blk in all_blocks:
        mask = 0
        for ts in itertools.combinations(blk, t):
            mask |= (1 << t_sub_to_bit[ts])
        block_masks.append(mask)
    
    # Initial uncovered mask: all bits set
    full_mask = (1 << n_t_subs) - 1
    uncovered = full_mask
    used = [False] * len(all_blocks)
    selected_indices = []
    
    # Helper for popcount
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    # Greedy selection
    while uncovered:
        best_idx = -1
        best_count = 0
        
        for i in range(len(all_blocks)):
            if used[i]:
                continue
            
            new_cover = block_masks[i] & uncovered
            count = popcount(new_cover)
            
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        used[best_idx] = True
        selected_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # Reduction phase
    # Try to remove blocks from the end (last added are often least critical)
    # We check if the remaining blocks still cover everything.
    # To do this efficiently, we can maintain a "coverage mask" of the current selection.
    # But removing one requires recomputing the union of the rest?
    # Or we can just check: 
    # Let M be the mask of selected blocks.
    # If we remove block j, new_mask = M - mask_j? No, masks overlap.
    # We need to know if any t-subset was covered ONLY by block j.
    # This is equivalent to: (M_without_j) == full_mask.
    
    # Efficient check:
    # Precompute prefix/suffix unions? 
    # Since N is small (~150), O(N^2) is fine.
    
    n_sel = len(selected_indices)
    if n_sel > 0:
        # Create a list of masks for selected blocks
        sel_masks = [block_masks[i] for i in selected_indices]
        
        # Try removing each block
        # We iterate backwards to potentially remove more
        for i in range(n_sel - 1, -1, -1):
            # Check if removing sel_masks[i] still covers full_mask
            # Compute union of all other masks
            union_others = 0
            for j in range(n_sel):
                if j != i:
                    union_others |= sel_masks[j]
            
            if union_others == full_mask:
                # Block i is redundant, remove it
                selected_indices.pop(i)
                sel_masks.pop(i)
                n_sel -= 1

    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in selected_indices]
