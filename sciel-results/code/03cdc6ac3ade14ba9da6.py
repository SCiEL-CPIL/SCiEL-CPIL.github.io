import itertools
import random
import time

def construct(v, k, t):
    # 1. Precompute all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Precompute all k-subsets and their bitmask of covered t-subsets
    k_subs = list(itertools.combinations(range(v), k))
    block_masks = []
    for k_sub in k_subs:
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            mask |= (1 << t_sub_to_idx[t_sub])
        block_masks.append(mask)
        
    all_mask = (1 << len(t_subs)) - 1
    
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    # Use 110s to be safe, leaving time for final conversion
    deadline = start_time + 110.0
    
    # Run multiple greedy constructions with random tie-breaking
    # We can use a counter to seed random differently each iteration
    run_id = 0
    
    while time.time() < deadline:
        random.seed(run_id)
        run_id += 1
        
        uncovered = all_mask
        blocks = []
        chosen = set()
        
        # Greedy Phase
        while uncovered:
            best_gain = 0
            candidates = []
            
            # Iterate through all possible blocks
            for i in range(len(k_subs)):
                if i in chosen:
                    continue
                
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered is not 0
                
            # Pick random from candidates to break ties
            best_block = random.choice(candidates)
            chosen.add(best_block)
            blocks.append(best_block)
            uncovered &= ~block_masks[best_block]
        
        # Deletion Phase
        # Remove redundant blocks
        # A block is redundant if its mask is a subset of the OR of all other blocks' masks.
        changed = True
        while changed:
            changed = False
            new_blocks = []
            # To make this faster, we can compute the total OR once and subtract?
            # No, we need OR of *others*.
            # N is small (~300), so O(N^2) is fine.
            for b in blocks:
                others_or = 0
                for b2 in blocks:
                    if b2 != b:
                        others_or |= block_masks[b2]
                
                # If block b covers any subset not covered by others, it's essential
                if (block_masks[b] & ~others_or) != 0:
                    new_blocks.append(b)
                else:
                    changed = True
            blocks = new_blocks
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
    # Convert indices back to actual blocks
    result = []
    for idx in best_blocks:
        result.append(list(k_subs[idx]))
        
    return result
