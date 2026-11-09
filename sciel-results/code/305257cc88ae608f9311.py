import time

def construct():
    prescribed = [1, 2, 5, 11]
    # Sort prescribed
    prescribed.sort()
    
    # Initial sums
    initial_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            initial_sums.add(prescribed[i] + prescribed[j])
            
    best_set = list(prescribed)
    
    # We want to maximize size. Max possible is 20.
    # We try to find a set of size K containing the prescribed ones.
    # We use backtracking.
    
    def is_valid(candidate, current_set, current_sums):
        for y in current_set:
            s = candidate + y
            if s in current_sums:
                return False
        if 2 * candidate in current_sums:
            return False
        return True

    def backtrack(current_set, current_sums, start_idx, target_size, time_limit):
        nonlocal best_set
        if time.time() > time_limit:
            return
        
        if len(current_set) == target_size:
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            return
        
        # Pruning: if we can't reach target_size with remaining elements
        # The candidates are from start_idx to 284.
        # Remaining candidates count is 284 - start_idx + 1.
        if (284 - start_idx + 1) < (target_size - len(current_set)):
            return

        # Try to add elements from start_idx to 284
        for x in range(start_idx, 285):
            if x in current_set:
                continue
            if is_valid(x, current_set, current_sums):
                # Add x
                new_sums = set(current_sums)
                for y in current_set:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                
                current_set.append(x)
                backtrack(current_set, new_sums, x + 1, target_size, time_limit)
                current_set.pop()
                
                # If we found a solution of this size, we can return immediately 
                # if we are just checking existence, but we want the actual set.
                # Since we are iterating K downwards, the first found is the max.
                if len(best_set) == target_size:
                    return

    # Try K from 20 down to len(prescribed)
    # To avoid timeout, we allocate time for each K
    total_time_limit = 110  # Leave 10s buffer
    start_time = time.time()
    
    for k in range(20, len(prescribed) - 1, -1):
        # Estimate time per k. Maybe 20s for 20, 5s for 19, etc?
        # Or just run until time runs out?
        # If we find a set of size k, we return it.
        # We need to ensure we don't spend 110s on k=20 and miss k=19.
        
        # Let's give 10 seconds for k=20, 5 for k=19, 5 for k=18...
        time_slice = 10 if k == 20 else 5
        local_limit = time.time() + time_slice
        
        # Reset best_set for this specific target? No, we want the global best.
        # But if we find size 20, we stop.
        
        # We need a fresh start for the backtracking for each K?
        # Yes, because the path to size K might be different.
        # However, the prescribed elements are fixed.
        
        current_set = list(prescribed)
        current_sums = set(initial_sums)
        
        # The backtracking function above modifies best_set.
        # We need to be careful with state.
        
        # Let's wrap it in a way that stops if found.
        
        # Optimization: If k is large, the search space is huge.
        # Simple backtracking might be too slow for K=20 in 10s.
        
        # Alternative: Use the greedy result as a lower bound.
        # Greedy result size is likely 18 or 19.
        
        # Let's try a hybrid:
        # 1. Run Greedy to get a baseline.
        # 2. Run Backtracking to improve.
        
    # Let's implement the Greedy first to have a baseline.
    a = list(prescribed)
    sums = set(initial_sums)
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
    best_set = a
    
    # Now try to improve with Backtracking
    # We try to find a set of size len(best_set) + 1
    # If found, update best_set and try for +2, etc.
    
    current_size = len(best_set)
    # We know max is 20.
    for target_size in range(current_size + 1, 21):
        found = False
        limit = time.time() + 10 # 10 seconds per attempt
        
        # Backtracking to find ANY set of size target_size
        # We can use the existing best_set as a seed? No, backtracking from scratch is cleaner but slower.
        # We can use "Restart" or "Local Search".
        
        # Let's use a simpler Backtracking that starts from prescribed and tries to reach target_size.
        # To speed up, we can use the heuristic: try numbers that are in the current best_set first?
        # No, that biases it.
        
        # Let's stick to the simple backtracking but with a time cut-off.
        
        def bt(current_set, current_sums, start_idx, target_size, limit_time):
            if time.time() > limit_time:
                return None
            if len(current_set) == target_size:
                return list(current_set)
            
            # Pruning
            if (284 - start_idx + 1) < (target_size - len(current_set)):
                return None

            for x in range(start_idx, 285):
                if x in current_set:
                    continue
                # Check validity
                valid = True
                for y in current_set:
                    if (x + y) in current_sums:
                        valid = False
                        break
                if valid and (2 * x) in current_sums:
                    valid = False
                
                if valid:
                    new_sums = set(current_sums)
                    for y in current_set:
                        new_sums.add(x + y)
                    new_sums.add(2 * x)
                    
                    current_set.append(x)
                    res = bt(current_set, new_sums, x + 1, target_size, limit_time)
                    current_set.pop()
                    
                    if res:
                        return res
            return None
        
        res = bt(list(prescribed), set(initial_sums), 12, target_size, limit)
        if res:
            best_set = res
            current_size = target_size
        else:
            # If we couldn't find target_size, we can stop?
            # Not necessarily, maybe a different path for target_size+1? No, if we can't find size K, we can't find size K+1.
            # So we break.
            break
            
    return best_set
