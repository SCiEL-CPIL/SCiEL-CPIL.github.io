import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    
    # 1. Map t-subsets to indices
    t_subsets = list(itertools.combinations(all_points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # 2. Precompute masks for all k-subsets
    # A mask is an integer where bit i is 1 if t_subsets[i] is a subset of the k-block.
    k_subsets = list(itertools.combinations(all_points, k))
    block_masks = []
    block_list = [] # Keep the actual subsets for returning
    
    for block in k_subsets:
        mask = 0
        # For each t-subset in this block, set the bit
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        block_list.append(block)
        
    n_blocks = len(k_subsets)
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = (1 << n_t_subsets) - 1
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            # Iterate through all blocks to find max gain
            for i in range(n_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
            
            # Randomly pick from best candidates to diversify
            if best_indices:
                chosen_idx = rng.choice(best_indices)
                selected_indices.append(chosen_idx)
                uncovered &= ~block_masks[chosen_idx]
                
            else:
                # Should not happen if problem is valid
                break
                
        return [block_list[i] for i in selected_indices]
        
    # Run multiple times to find best
    best_blocks = None
    best_len = float('inf')
    start_time = time.time()
    time_limit = 115 # Leave buffer
    
    seed = 42
    while time.time() - start_time < time_limit:
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # Optional: print progress
            # print(f"Seed {seed}: {best_len} blocks")
        seed += 1
        
    return best_blocks
