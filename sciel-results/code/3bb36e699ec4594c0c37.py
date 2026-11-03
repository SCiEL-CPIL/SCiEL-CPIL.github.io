import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # 1. Enumerate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Number of 64-bit words needed to represent num_t_subs bits
    num_words = (num_t_subs + 63) // 64
    
    # 2. Enumerate all k-subsets and compute their coverage masks
    k_subs = list(itertools.combinations(range(v), k))
    num_k_subs = len(k_subs)
    
    # masks[i] is a numpy array of num_words 64-bit integers
    masks = np.zeros((num_k_subs, num_words), dtype=np.uint64)
    
    for i, k_sub in enumerate(k_subs):
        k_set = set(k_sub)
        # Find all t-subsets contained in k_sub
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_index[t_sub]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks[i, word_idx] |= (1 << bit_idx)
            
    # Check time
    if time.time() > deadline:
        # Fallback to simple greedy if precomputation is too slow (unlikely)
        uncovered = set(t_subs)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    # 3. Greedy Set Cover
    # uncovered_mask starts as all 1s (all bits set)
    # We represent the mask as an array of uint64
    uncovered = np.ones(num_words, dtype=np.uint64)
    # Mask out bits beyond num_t_subs in the last word
    last_word_bits = num_t_subs % 64
    if last_word_bits > 0:
        uncovered[-1] = (1 << last_word_bits) - 1
    
    selected_blocks = []
    
    while np.any(uncovered):
        # Calculate gain for each block: popcount(masks[i] & uncovered)
        # We can do this word by word, but numpy doesn't have popcount for uint64 directly in all versions?
        # Actually, np.bitwise_and is fast. Popcount is the bottleneck.
        # We can use a lookup table for popcount of 16-bit chunks or similar.
        # Or, since num_words is ~183, and num_k_subs is ~75k, that's 13.5 million operations.
        # Let's try a simple loop over words for each block? No, that's slow in Python.
        
        # Alternative: Use a precomputed popcount table for 16-bit values.
        # But numpy vectorization is key.
        # Let's just use a simple heuristic: 
        # Actually, for this specific size, a pure Python loop with bit operations on integers might be faster than numpy overhead if we convert to int.
        # But 75k blocks * 183 words is too big for a single Python int? No, Python ints are arbitrary precision.
        
        # Let's switch to Python integers for masks. It's simpler and often fast enough for this size.
        pass # We'll stick to the numpy plan but optimize the gain calculation.
        
    # Re-implementation with Python Ints for masks is safer for portability and speed at this scale.
    # Let's restart the core logic with Python Ints.
    
    if time.time() > deadline:
         # This branch is unlikely to be hit if we are fast, but good for safety.
         return []

    # --- Python Int Mask Implementation ---
    # Re-compute masks as Python ints
    masks_int = [0] * num_k_subs
    for i, k_sub in enumerate(k_subs):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_index[t_sub]
            mask |= (1 << idx)
        masks_int[i] = mask
        
    if time.time() > deadline:
        # Fallback
        return []

    uncovered_int = (1 << num_t_subs) - 1
    selected_indices = []
    
    # Pre-sort or just scan all? Scanning 75k items 300 times is 22.5M iterations.
    # In Python, this might be ~5-10 seconds.
    # We can optimize by maintaining a list of active candidates.
    
    active = list(range(num_k_subs))
    
    while uncovered_int:
        best_idx = -1
        best_gain = -1
        
        # Check time frequently
        if time.time() > deadline - 5:
            break
            
        # Find best block
        # To speed up, we can sample or use a heuristic.
        # But 75k is small. Let's just scan.
        for idx in active:
            gain = (masks_int[idx] & uncovered_int).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if gain == 0:
                    # If best gain is 0, we are stuck (should not happen if uncovered is not 0)
                    break
            # Optimization: if best_gain is max possible (56), we can't do better
            if best_gain == 56: 
                # We can't gain more than 56 per block. If we hit 56, it's optimal for this step.
                # But we still need to check if there are other 56s? No, any 56 is equally good.
                # We can break early if we find a perfect cover? No, we just pick one.
                pass

        if best_idx == -1 or best_gain == 0:
            # Should not happen if uncovered is not 0
            break
            
        selected_indices.append(best_idx)
        uncovered_int &= ~masks_int[best_idx]
        
        # Remove blocks that are now completely covered? No, they might still be useful?
        # Actually, if a block's mask is a subset of the *covered* part, it's useless for future gains?
        # No, gain is calculated on *uncovered*.
        # If a block covers only already covered bits, its gain is 0.
        # We can remove blocks with gain 0 from active list to speed up future iterations.
        # But calculating gain for all active blocks every step is the cost.
        
        # Optimization: Only keep blocks that have non-zero intersection with uncovered.
        # We can filter active list.
        # new_active = [idx for idx in active if (masks_int[idx] & uncovered_int) != 0]
        # This is O(N) per step.
        
        # Let's just scan all active.
        
    # Convert indices to blocks
    blocks = [list(k_subs[i]) for i in selected_indices]
    
    # 4. Post-processing: Remove redundant blocks
    # A block is redundant if all its 5-subsets are covered by other blocks.
    # We need to check coverage for each 5-subset in the block.
    
    # Build a map: for each 5-subset, how many blocks cover it?
    # Or simpler: 
    # For each block B in selected_blocks:
    #   For each 5-subset S in B:
    #     Check if S is covered by any other block B' != B.
    
    # This is O(N_blocks * 56 * N_blocks). ~300 * 56 * 300 = 5M checks.
    # Each check is a set membership or bitmask check.
    
    # Let's use bitmasks again for this.
    # coverage_count[s_idx] = number of blocks covering s_idx.
    # Or just a boolean array covered_by_other?
    
    # Efficient way:
    # 1. Create an array `cover_count` of size num_t_subs, initialized to 0.
    # 2. For each block in selected_blocks, increment count for all its 5-subsets.
    # 3. A 5-subset is "uniquely covered" by block B if cover_count[s_idx] == 1 and B covers it.
    # 4. A block B is redundant if it has NO uniquely covered 5-subsets.
    
    # Let's implement this.
    
    if time.time() > deadline - 2:
        return blocks
        
    cover_count = np.zeros(num_t_subs, dtype=np.int32)
    
    # Map block indices to their 5-subset indices for faster access
    block_t_subs = []
    for i, k_sub in enumerate(k_subs):
        t_subs_in_block = [t_sub_index[s] for s in itertools.combinations(k_sub, t)]
        block_t_subs.append(t_subs_in_block)
        
    # Update counts
    for idx in selected_indices:
        for s_idx in block_t_subs[idx]:
            cover_count[s_idx] += 1
            
    # Check for redundancy
    final_selected = []
    for idx in selected_indices:
        is_redundant = True
        for s_idx in block_t_subs[idx]:
            if cover_count[s_idx] == 1:
                # This subset is only covered by this block, so block is essential
                is_redundant = False
                break
        if not is_redundant:
            final_selected.append(idx)
            
    # We might have removed some blocks, which makes other blocks redundant.
    # Iterate until stable.
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        # Recalculate cover counts for current final_selected
        cover_count = np.zeros(num_t_subs, dtype=np.int32)
        for idx in final_selected:
            for s_idx in block_t_subs[idx]:
                cover_count[s_idx] += 1
                
        new_final = []
        for idx in final_selected:
            is_redundant = True
            for s_idx in block_t_subs[idx]:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_final.append(idx)
            else:
                changed = True
                
        final_selected = new_final
        
    blocks = [list(k_subs[i]) for i in final_selected]
    
    return blocks
