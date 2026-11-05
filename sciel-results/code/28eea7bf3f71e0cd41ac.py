import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute bitsets for all possible k-blocks
    # A block is a k-subset. It covers all t-subsets that are subsets of it.
    # We represent the set of covered t-subsets as a Python int bitset.
    k_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(k_blocks)}
    
    # For each k-block, compute the bitset of t-subsets it covers
    # This is expensive to do for all k_blocks if v is large, but for v=19, k=8, 
    # C(19,8) = 75582, which is manageable.
    # However, we don't need to precompute ALL k_blocks bitsets upfront if it's too slow.
    # Instead, we can compute the bitset on the fly or precompute a mapping from 
    # (t-subset) to the list of k-blocks that contain it.
    
    # Let's use the approach: 
    # 1. Maintain a bitset `uncovered` of uncovered t-subsets.
    # 2. For a candidate block, compute how many uncovered t-subsets it covers.
    # 3. To speed this up, precompute for each t-subset, the list of k-blocks containing it? 
    #    No, that's C(19,5) * C(19-5, 8-5) = 11628 * C(14,3) = 11628 * 364 ~ 4M entries. Too big.
    
    # Alternative: Precompute the bitset for each k-block.
    # C(19,8) = 75582. Each bitset is an int. This is feasible.
    
    block_bitsets = {}
    for b in k_blocks:
        mask = 0
        # For each t-subset in b, set the bit
        for ts in itertools.combinations(b, t):
            idx = subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitsets[b] = mask
        
    # Sort blocks by size of coverage? Not needed, greedy picks best at each step.
    
    # Greedy Construction
    uncovered_mask = (1 << n_t_subsets) - 1
    chosen_blocks = []
    
    while uncovered_mask:
        # Check time
        if time.time() > deadline:
            break
        
        # Find the block that covers the most uncovered t-subsets
        best_block = None
        best_count = -1
        # To speed up, we can sample or use a heuristic. But for v=19, 75k blocks is okay.
        # We can optimize by only considering blocks that contain the first uncovered t-subset.
        
        # Find the first uncovered t-subset
        # Get index of least significant set bit
        lsb_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        first_uncovered_ts = t_subsets[lsb_idx]
        
        # Only consider blocks that contain first_uncovered_ts
        # Precompute: for each t-subset, which k-blocks contain it?
        # Let's precompute this mapping to speed up the greedy step.
        # ts_to_blocks: dict from t-subset index to list of block indices
        ts_to_blocks = [[] for _ in range(n_t_subsets)]
        for i, b in enumerate(k_blocks):
            b_set = set(b)
            for ts in itertools.combinations(b, t):
                idx = subset_to_idx[ts]
                ts_to_blocks[idx].append(i)
        
        # Now, for the current first_uncovered_ts, we only check blocks in ts_to_blocks[lsb_idx]
        candidate_blocks = ts_to_blocks[lsb_idx]
        
        for bi in candidate_blocks:
            b = k_blocks[bi]
            mask = block_bitsets[b]
            count = (uncovered_mask & mask).bit_count()
            if count > best_count:
                best_count = count
                best_block = b
        
        if best_block is None:
            break
            
        chosen_blocks.append(best_block)
        uncovered_mask &= ~block_bitsets[best_block]
        
    # Local Search: Block Exchange
    # Try to replace two blocks with one if possible.
    # We iterate until no improvement is found or time runs out.
    
    # To make local search efficient, we need to quickly check if the union of two blocks' 
    # covered t-subsets is covered by a single k-block.
    # Union mask = mask1 | mask2
    # We need to find if there exists a k-block B such that (mask1 | mask2) & ~mask_B == 0.
    # i.e., mask1 | mask2 is a subset of mask_B.
    
    # Precompute a mapping from mask to block? No, masks are large.
    # Instead, for a given union mask U, we can check all k-blocks? That's 75k checks per pair.
    # Number of pairs is C(N,2) where N is number of blocks (~300). 300*299/2 ~ 45k pairs.
    # 45k * 75k = 3.3 billion checks. Too slow.
    
    # Better: For each k-block B, it can cover any union U that is a subset of mask_B.
    # We can iterate over k-blocks B, and check if there are two blocks in chosen_blocks 
    # whose union is a subset of mask_B.
    
    # Let's try a different local search: 
    # Remove a block if its covered t-subsets are covered by the remaining blocks.
    # This is simpler and faster.
    
    for _ in range(100): # Max iterations for local search
        if time.time() > deadline:
            break
            
        improved = False
        # Try removing each block
        for i in range(len(chosen_blocks)):
            if time.time() > deadline:
                break
                
            b_i = chosen_blocks[i]
            mask_i = block_bitsets[b_i]
            
            # Check if mask_i is covered by the union of all other blocks
            # We can compute the union of all other blocks' masks
            # This is O(N) per block, so O(N^2) total.
            # N is ~300, so 90k operations. Very fast.
            
            # To speed up, we can maintain a global union mask and subtract? 
            # But we need to check if mask_i is subset of (GlobalUnion - mask_i + mask_i)? 
            # No, we need to check if mask_i is subset of (Union of all j != i).
            
            # Let's compute union of all j != i
            # We can precompute prefix/suffix unions? Or just do it.
            # Since N is small, just do it.
            
            other_union = 0
            for j in range(len(chosen_blocks)):
                if j != i:
                    other_union |= block_bitsets[chosen_blocks[j]]
                    
            if (mask_i & ~other_union) == 0:
                # Block i is redundant
                chosen_blocks.pop(i)
                improved = True
                break # Restart loop as indices changed
        
        if not improved:
            # Try block exchange: replace two blocks with one
            # This is more complex. Let's stick to removal first.
            # If no removal possible, try exchange.
            
            # Exchange: Find i, j such that mask_i | mask_j is subset of mask_k for some k not in chosen_blocks
            # This is expensive. Let's skip for now and see if removal is enough.
            # Given the time constraint, removal is a safe and effective first step.
            
            # If we have time, we could try a random restart or a more sophisticated exchange.
            # But for now, let's assume removal gets us close to optimal.
            break

    # Convert back to lists of ints
    # The blocks are already tuples of ints.
    return [list(b) for b in chosen_blocks]
