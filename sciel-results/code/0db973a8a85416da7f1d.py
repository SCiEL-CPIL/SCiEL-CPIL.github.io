import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    points = range(v)
    t_subs = list(itertools.combinations(points, t))
    num_t_subs = len(t_subs)
    
    k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(k_blocks)
    
    # Precompute bitmask for each k-block
    block_masks = [0] * num_k_blocks
    for b_idx, block in enumerate(k_blocks):
        block_set = set(block)
        mask = 0
        for t_idx, t_sub in enumerate(t_subs):
            # Check if t_sub is subset of block
            # Since t is small (5) and k is small (8), this is fast
            if all(p in block_set for p in t_sub):
                mask |= (1 << t_idx)
        block_masks[b_idx] = mask
        
    best_blocks = []
    best_count = float('inf')
    
    # Use randomization to explore different greedy paths
    # We can randomize the order of blocks considered or the tie-breaking
    for restart in range(50):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << num_t_subs) - 1
        current_blocks = []
        
        while uncovered_mask:
            max_gain = 0
            best_block_indices = []
            
            # Find all blocks with max gain
            # To add randomness, we can pick randomly from top candidates
            gains = [0] * num_k_blocks
            max_gain = 0
            for b_idx in range(num_k_blocks):
                gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                gains[b_idx] = gain
                if gain > max_gain:
                    max_gain = gain
                    best_block_indices = [b_idx]
                elif gain == max_gain and max_gain > 0:
                    best_block_indices.append(b_idx)
            
            if max_gain == 0:
                # No block covers any new subset? Error
                break
                
            # Randomly pick from best_block_indices to diversify
            # Or just pick the first one? 
            # Let's pick randomly from the top 10% of gains if available, 
            # or just from the max gain set.
            # Simple: pick randomly from best_block_indices
            chosen_idx = random.choice(best_block_indices)
            
            current_blocks.append(k_blocks[chosen_idx])
            uncovered_mask &= ~block_masks[chosen_idx]
            
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    return [list(b) for b in best_blocks]
