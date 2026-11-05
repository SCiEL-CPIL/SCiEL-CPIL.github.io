import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts if time permits
    # The greedy is deterministic if we use min(uncovered).
    # To vary, we can shuffle the order of candidates or the seed selection.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        uncovered = set(range(len(all_t_subs)))
        blocks = []
        
        # Shuffle the list of subsets for seed selection?
        # Or just use min.
        # Let's try a randomized seed selection for diversity in restarts.
        # But min is a good heuristic (covers "leftmost" holes).
        # Let's stick to min for the first run, then random for others.
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Select seed
            if num_restarts == 1:
                seed_idx = min(uncovered)
            else:
                seed_idx = random.choice(list(uncovered))
            
            seed_sub = all_t_subs[seed_idx]
            block = list(seed_sub)
            block_set = set(block)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                # To break ties, we can use a random tie-breaker or index
                best_x_idx = -1
                
                for x in all_points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Iterate (t-1) subsets of block
                    # block is a list. We need sorted combinations.
                    # Since block is built by adding elements, it's not necessarily sorted.
                    # We can keep block sorted.
                    # Let's maintain block as a sorted list.
                    
                    # Actually, let's just sort block for combinations.
                    # Since k is small (7), sorting is cheap.
                    sorted_block = sorted(block)
                    for r in itertools.combinations(sorted_block, t - 1):
                        # Insert x into r to form cand
                        # r is a tuple of t-1 sorted ints
                        # x is an int
                        # Create sorted tuple
                        # Optimization: manual insertion
                        cand_list = list(r)
                        # Find position for x
                        # Since r is sorted, we can use bisect
                        import bisect
                        pos = bisect.bisect_left(cand_list, x)
                        cand_list.insert(pos, x)
                        cand = tuple(cand_list)
                        
                        if cand in sub_to_idx:
                            idx = sub_to_idx[cand]
                            if idx in uncovered:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_x_idx = x
                    elif gain == best_gain:
                        # Tie break: smaller x? or random?
                        if num_restarts > 1:
                            if random.random() < 0.5:
                                best_x = x
                                best_x_idx = x
                        else:
                            if x < best_x:
                                best_x = x
                                best_x_idx = x
                                
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
                
            if len(block) == k:
                blocks.append(sorted(block))
                
                # Remove covered
                for sub in itertools.combinations(sorted(block), t):
                    if sub in sub_to_idx:
                        idx = sub_to_idx[sub]
                        if idx in uncovered:
                            uncovered.remove(idx)
                            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we found a very good one (e.g. <= 151), we might stop early?
        # No, try to improve.
        # But if best_len is 151, it's hard to beat.
        
        # Limit restarts to avoid infinite loop if time is not checked properly
        if num_restarts > 100:
            break

    return best_blocks if best_blocks is not None else []
