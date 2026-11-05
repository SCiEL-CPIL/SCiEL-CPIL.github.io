import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Precompute all t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    combo_to_idx = {c: i for i, c in enumerate(t_subsets)}
    
    # 2. Precompute all k-blocks and their masks
    # A block covers a t-subset if the t-subset is a subset of the block.
    # Mask for a block B is a bitvector where bit i is 1 if t_subsets[i] is subset of B.
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_mask = {}
    for b in all_blocks:
        mask = 0
        # Check all t-subsets of b
        for ts in itertools.combinations(b, t):
            mask |= (1 << combo_to_idx[ts])
        block_to_mask[b] = mask
        
    # 3. Greedy Initialization
    uncovered_mask = (1 << n_t_subsets) - 1
    selected_blocks = []
    selected_masks = []
    
    # To speed up greedy, we can sort blocks by potential gain, but gain changes.
    # We just iterate.
    while uncovered_mask and time.time() < deadline - 20: # Reserve 20s for local search
        best_gain = -1
        best_block = None
        best_mask = 0
        
        # Optimization: Only consider blocks that have non-zero intersection with uncovered
        # But checking all is O(N_blocks).
        for b in all_blocks:
            m = block_to_mask[b]
            gain = (uncovered_mask & m).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b
                best_mask = m
        
        if best_gain == 0:
            break # Should not happen if covering is possible
            
        selected_blocks.append(best_block)
        selected_masks.append(best_mask)
        uncovered_mask &= ~best_mask
        
    if not selected_blocks:
        return []

    # 4. Local Search: Remove Redundant Blocks
    # We repeat this phase because removing one block might make another redundant.
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        new_blocks = []
        new_masks = []
        for i, b in enumerate(selected_blocks):
            # Check if this block is redundant
            # Redundant if (union of all other masks) covers (mask of this block)
            # i.e., (mask_i & ~(OR of others)) == 0
            
            # Efficient check:
            # Calculate OR of all masks except i? That's O(N) per block, total O(N^2).
            # N is ~300. 300*300 = 90k operations. Very fast.
            
            others_union = 0
            for j, m in enumerate(selected_masks):
                if i != j:
                    others_union |= m
            
            if (selected_masks[i] & ~others_union) == 0:
                # Redundant, skip adding to new lists
                changed = True
            else:
                new_blocks.append(b)
                new_masks.append(m)
        
        selected_blocks = new_blocks
        selected_masks = new_masks

    # 5. Local Search: Point Swapping (Improvement)
    # Try to improve individual blocks to cover "critical" subsets better.
    # Critical subset: one covered by very few blocks.
    # Heuristic: For each block, try swapping a point x in block with y not in block.
    # Accept if the new block covers more subsets that are currently covered by ONLY the old block?
    # Or simpler: Accept if the new block covers more uncovered subsets (if we allow gaps temporarily? No).
    # Better: Accept if the new block covers more subsets that are "rare".
    
    # Let's calculate coverage counts for each t-subset.
    # This is expensive to recompute every step.
    # Alternative: Just try to swap if it doesn't break coverage.
    # If we swap, we must ensure no t-subset loses all coverage.
    
    # Given time constraints, a simple "try all swaps, check validity" might be too slow if done naively.
    # Let's try a limited number of random restarts or local moves.
    
    # Since we already have a valid covering from Greedy + Redundancy Removal,
    # and the main gain comes from Redundancy Removal (which is very effective),
    # we will focus on ensuring the Redundancy Removal is thorough.
    
    # One more pass of redundancy removal with a different order?
    # The order of removal doesn't matter for the final set of non-redundant blocks if we iterate to fixpoint.
    
    # Let's add a "Swap" phase that is conservative.
    # For each block, try to replace it with another block that covers the same "critical" parts.
    # Actually, let's just return the result of Greedy + Redundancy Removal.
    # This is often close to optimal for small v.
    
    # Convert blocks to lists of ints
    return [list(b) for b in selected_blocks]
