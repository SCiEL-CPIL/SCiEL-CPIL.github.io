import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    masks = []
    for block in k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        masks.append(mask)
        
    # Initial full mask
    full_mask = (1 << num_t_subs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple times with random tie-breaking
    while time.time() < deadline:
        uncovered_mask = full_mask
        selected_indices = []
        
        while uncovered_mask:
            # Find max gain
            max_gain = 0
            candidates = []
            
            # Optimization: If we find a gain of num_t_subs, we can break early?
            # Max possible gain is num_t_subs.
            # But usually much less.
            
            for i, m in enumerate(masks):
                g = (m & uncovered_mask).bit_count()
                if g > max_gain:
                    max_gain = g
                    candidates = [i]
                elif g == max_gain and g > 0:
                    candidates.append(i)
                    
            if max_gain == 0:
                break # Should not happen if uncovered_mask != 0
                
            # Random choice among candidates
            best_idx = random.choice(candidates)
            selected_indices.append(best_idx)
            uncovered_mask &= ~masks[best_idx]
            
        current_count = len(selected_indices)
        if current_count < best_count:
            best_count = current_count
            best_blocks = [list(k_blocks[i]) for i in selected_indices]
            
        # If we found a very good solution, we can stop early?
        # Theoretical lower bound is 264. Best known 335.
        # If we get <= 335, it's great.
        if best_count <= 335:
             # We might try to improve further, but 335 is the target.
             # Let's continue a bit longer to see if we can go lower.
             pass

    return best_blocks
