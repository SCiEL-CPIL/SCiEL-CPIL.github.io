import time

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = 80  # safety margin
    
    # Precompute inverses and pair IDs
    # pair_id[x] is the ID of the pair containing x
    # Pairs are {x, inv[x]}
    # Self-inverses: 1 and 1008.
    # We assign unique IDs to all pairs.
    
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    pair_id = [-1] * (MOD)
    pair_counter = 0
    is_self_inverse = [False] * (MOD)
    
    for x in range(1, MOD):
        if pair_id[x] != -1:
            continue
        y = inv[x]
        if x == y:
            # Self-inverse
            pair_id[x] = pair_counter
            is_self_inverse[pair_counter] = True
            pair_counter += 1
        else:
            # Non-self-inverse pair
            pair_id[x] = pair_counter
            pair_id[y] = pair_counter
            is_self_inverse[pair_counter] = False
            pair_counter += 1
            
    max_pairs = pair_counter
    counts = [0] * max_pairs
    
    best_start = 1
    best_len = 0
    L = 1
    
    for R in range(1, MOD):
        if time.time() - start_time > deadline:
            break
            
        current_id = pair_id[R]
        counts[current_id] += 1
        
        # Shrink L if the window is invalid
        # Invalid if:
        # 1. counts[current_id] > 1 (for non-self-inverse)
        # 2. counts[current_id] > 0 (for self-inverse)
        # Actually, we need to check if ANY pair has too many elements.
        # But only the pair of R could have changed its count to become invalid.
        # Previous pairs were valid.
        # So we only need to check current_id.
        
        while True:
            if is_self_inverse[current_id]:
                if counts[current_id] > 0:
                    # Must remove current_id entirely
                    left_id = pair_id[L]
                    counts[left_id] -= 1
                    L += 1
                    # After removing, check if the NEW current_id (which is still the one we added? No, we added R, so current_id is fixed for this R)
                    # Wait, if we shrink L, we might remove elements of OTHER pairs.
                    # But the condition for validity is global.
                    # However, only the pair `current_id` was incremented.
                    # If `current_id` is self-inverse, we need its count to be 0.
                    # We keep shrinking L until counts[current_id] becomes 0.
                    continue
                else:
                    break
            else:
                if counts[current_id] > 1:
                    left_id = pair_id[L]
                    counts[left_id] -= 1
                    L += 1
                    continue
                else:
                    break
                    
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
            
    return [best_start, best_len]
