import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    uncovered = set(range(n_subsets))
    blocks = []
    
    # Precompute 4-combinations for gain calculation?
    # No, block changes.
    
    while uncovered:
        if time.time() - start_time > 100:
            # Emergency break? But must return valid.
            # If we break here, uncovered is not empty.
            # We must finish.
            pass
            
        # Pick an uncovered subset
        # Using min() is O(N) but N=11628, fine.
        # Or just pop an arbitrary one? set.pop() is O(1).
        # But we need to know which one it is to start the block.
        # Let's pick the one with the smallest ID for determinism.
        first_uncovered_id = min(uncovered)
        first_uncovered_subset = all_t_subsets[first_uncovered_id]
        
        block = list(first_uncovered_subset)
        
        while len(block) < k:
            block_set = set(block)
            # Precompute (t-1)-combinations of current block
            t_minus_1_combos = list(itertools.combinations(block, t - 1))
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for combo in t_minus_1_combos:
                    # Create the t-subset containing x and combo
                    # combo is a tuple of t-1 elements
                    # We need to check if sorted(combo + (x,)) is in uncovered
                    # Optimization: use frozenset for key
                    s_t = frozenset(combo) | {x}
                    sid = subset_to_id.get(s_t)
                    if sid is not None and sid in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for t_sub in itertools.combinations(block, t):
            sid = subset_to_id[t_sub]
            if sid in uncovered:
                uncovered.remove(sid)
                
    return blocks
