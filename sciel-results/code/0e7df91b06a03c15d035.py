import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # 1. Generate all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)
    
    # 3. Pre-compute masks for each k-block
    # mask[b] has bit i set if t_subs[i] is a subset of k_blocks[b]
    block_masks = [0] * n_k_blocks
    for b, block in enumerate(k_blocks):
        mask = 0
        # A k-block contains C(k, t) t-subsets
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_index[t_sub]
            mask |= (1 << idx)
        block_masks[b] = mask
        
    best_solution = None
    best_size = float('inf')
    
    # Run greedy algorithm multiple times with random tie-breaking
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_t_subs) - 1
        used_blocks = set()
        current_solution = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidates = []
            
            # Find the block(s) that cover the most uncovered t-subsets
            for b in range(n_k_blocks):
                if b in used_blocks:
                    continue
                # Calculate gain: number of newly covered t-subsets
                gain = (block_masks[b] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(b)
                    
            if not best_candidates:
                # Should not happen if uncovered_mask is not 0 and blocks exist
                break
                
            # Pick a random candidate to add diversity and escape local minima
            chosen_b = random.choice(best_candidates)
            
            used_blocks.add(chosen_b)
            current_solution.append(list(k_blocks[chosen_b]))
            uncovered_mask &= ~block_masks[chosen_b]
            
        # Check if we found a valid covering
        if uncovered_mask == 0:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = current_solution
                
            # If we have a very good solution, we might want to try harder to improve
            # but for now, we just record the best.
            if best_size <= 330:
                # We can break early if we hit a known good bound, 
                # but let's try to beat 335.
                pass

    if best_solution is None:
        # Fallback: return an empty list or raise error? 
        # The task requires a valid covering. 
        # If greedy failed, we might need a different strategy, 
        # but for C(19,8,5) greedy usually works.
        return []

    return best_solution
