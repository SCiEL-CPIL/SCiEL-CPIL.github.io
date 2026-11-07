    # Local Search
    current_set = best_set[:]
    
    while time.time() < deadline:
        # 1. Try to Add
        added = False
        candidates_to_add = [x for x in range(1, 257) if x not in current_set]
        random.shuffle(candidates_to_add)
        
        for x in candidates_to_add:
            if time.time() > deadline:
                break
            if is_valid_fast(current_set + [x]):
                current_set.append(x)
                added = True
                break
        
        if added:
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            continue
            
        # 2. Try to Swap (maintain size)
        swapped = False
        # Limit swaps per loop to avoid wasting time
        for _ in range(50):
            if time.time() > deadline:
                break
            if len(current_set) == 0:
                break
            x_out = random.choice(current_set)
            candidates_to_add = [x for x in range(1, 257) if x not in current_set]
            if not candidates_to_add:
                break
            x_in = random.choice(candidates_to_add)
            
            if is_valid_fast([x for x in current_set if x != x_out] + [x_in]):
                current_set.remove(x_out)
                current_set.append(x_in)
                swapped = True
                break
                
        if swapped:
            continue
            
        # 3. Stuck. Try to Remove or Restart
        # If we are at a local maximum for size k, we might need to drop to k-1 to find a path to k+1.
        # But dropping is expensive.
        # Let's try a few random restarts of the greedy phase if stuck.
        
        # Check if we should restart
        # Simple heuristic: if we haven't improved best_set in a while, restart.
        # We don't track "time since improvement" explicitly, but we can just restart every N loops if stuck.
        
        # For now, just break and restart greedy with different random seed?
        # Or just try removing one element and see if we can add two?
        
        # Let's try removing a random element
        if len(current_set) > 0:
            x_out = random.choice(current_set)
            current_set.remove(x_out)
        else:
            # Restart
            current_set = []
            
        # After removal, try to add
        # (Loop continues)

    return list(best_set)
