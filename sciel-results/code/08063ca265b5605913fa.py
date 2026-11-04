import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    index_to_comb = all_tsubsets
    comb_to_index = {c: i for i, c in enumerate(all_tsubsets)}
    
    # We need subset_mask[S] for point masks S with popcount(S) <= k
    # subset_mask[S] is an integer where bit i is set if all_tsubsets[i] is a subset of S
    
    # To save memory and time, we only compute for S with popcount <= k.
    # We store in a dict or list indexed by S? S can be up to 2^19.
    # List of size 2^v is 524288. This is fine.
    subset_mask = [0] * (1 << v)
    
    # Precompute point masks for each t-subset to help?
    # Actually, for each S, we find its points, then combinations.
    # Optimization: Only compute for S with popcount >= t and <= k.
    
    # Generate all masks S with popcount in [t, k]
    valid_masks = []
    for size in range(t, k + 1):
        for comb in itertools.combinations(range(v), size):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            valid_masks.append(mask)
            
    # Compute subset_mask for these masks
    # To speed up, we can use the fact that if S1 subset S2, mask[S1] subset mask[S2]
    # But direct calculation is fine for small v.
    
    # Cache for points in mask to avoid list comprehension overhead?
    # Let's just iterate.
    
    for S in valid_masks:
        # Get points in S
        points = []
        for p in range(v):
            if S & (1 << p):
                points.append(p)
        
        m = 0
        # Iterate t-subsets of points
        for tc in itertools.combinations(points, t):
            idx = comb_to_index[tc]
            m |= (1 << idx)
        subset_mask[S] = m
        
    # 2. Greedy Construction
    uncovered = (1 << n_tsubsets) - 1
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Find the first uncovered t-subset (lexicographically smallest index)
        # Least significant set bit
        lsb = uncovered & -uncovered
        idx = lsb.bit_length() - 1
        
        start_comb = index_to_comb[idx]
        
        # Initialize block with start_comb
        block_points = list(start_comb)
        block_mask = 0
        for p in block_points:
            block_mask |= (1 << p)
            
        # Greedily add points until size k
        while len(block_points) < k:
            best_gain = -1
            best_x = -1
            
            for x in range(v):
                if block_mask & (1 << x):
                    continue
                    
                new_block_mask = block_mask | (1 << x)
                
                # Get masks
                new_t_mask = subset_mask[new_block_mask]
                old_t_mask = subset_mask[block_mask]
                
                # Gain: subsets covered by new_block_mask but not old_block_mask, 
                # that are currently uncovered
                gain_mask = (new_t_mask & ~old_t_mask) & uncovered
                gain = gain_mask.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            # If best_gain is 0, we might still need to add a point to reach size k
            # but it won't cover anything new. 
            # However, we started with an uncovered set, so the first additions should have gain > 0.
            # If we get stuck, we pick the smallest x not in block.
            if best_x == -1:
                 # Fallback: pick any point not in block
                 for x in range(v):
                     if not (block_mask & (1 << x)):
                         best_x = x
                         break
                         
            block_points.append(best_x)
            block_mask |= (1 << best_x)
            
        block_points.sort()
        blocks.append(block_points)
        
        # Update uncovered
        final_mask = subset_mask[block_mask]
        uncovered &= ~final_mask
        
    return blocks
