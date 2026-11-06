import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin

    # 1. Generate Greedy Sequence
    values = []
    sums = set()
    prefix = [0]
    
    for x in range(1, 257):
        P_new = prefix[-1] + x
        new_diffs = [P_new - p for p in prefix]
        if not sums.intersection(new_diffs):
            values.append(x)
            sums.update(new_diffs)
            prefix.append(P_new)
            
    greedy_len = len(values)
    greedy_values = values[:]
    greedy_sums = sums.copy()
    greedy_prefix = prefix[:]

    # 2. Try to find a longer sequence (greedy_len + 1)
    # We will try to backtrack from the end of the greedy sequence.
    # This is a heuristic: maybe the last few choices were suboptimal 
    # for extending the length, even if they were minimal.
    
    # We define a recursive search function that tries to extend the sequence
    # from a given state (current values, sums, prefix) to a target length.
    
    target_len = greedy_len + 1
    
    # To make search tractable, we only allow backtracking in the last K steps.
    # If we backtrack earlier, the search space explodes.
    # Let's try K=5.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(current_values, current_sums, current_prefix, start_idx, target_len):
        if time.time() > deadline:
            return None
        
        if len(current_values) == target_len:
            return current_values[:]
        
        # If we are in the "fixed" prefix part (before backtracking zone), 
        # we just follow the greedy path or fail.
        # Actually, we want to modify the tail.
        
        # Let's try to find a valid x for the next position.
        last_val = current_values[-1] if current_values else 0
        last_prefix = current_prefix[-1]
        
        # Candidates must be > last_val and <= 256
        # To allow for more future steps, maybe we should prefer smaller x?
        # Greedy does this.
        # If greedy failed to reach target_len, it means at some step, 
        # no x <= 256 worked.
        # If we backtrack, we try a larger x at an earlier step, 
        # which might allow a smaller x later?
        # Or just allow a different path.
        
        # We iterate through possible x values.
        # To prune, we check if x is valid.
        
        for x in range(last_val + 1, 257):
            P_new = last_prefix + x
            new_diffs = [P_new - p for p in current_prefix]
            
            if current_sums.intersection(new_diffs):
                continue
                
            # If valid, recurse
            new_values = current_values + [x]
            new_sums = current_sums.union(new_diffs)
            new_prefix = current_prefix + [P_new]
            
            # Optimization: If we are in the backtracking zone, 
            # we can try all options. 
            # If we are outside, we should stick to greedy to avoid branching.
            
            res = search(new_values, new_sums, new_prefix, start_idx, target_len)
            if res:
                return res
                
            # If we are in the backtracking zone, we try next x.
            # If we are outside, we should break? 
            # No, if we are outside, we should only try the greedy x?
            # If we only try greedy x outside the zone, and it fails, we return None.
            
            # To keep it simple: 
            # We only backtrack in the last 5 steps.
            # If current index < len(greedy_values) - 5, we ONLY try the greedy value.
            
            if len(current_values) < greedy_len - 5:
                # Only try the greedy value
                if x != greedy_values[len(current_values)]:
                    break # Only one candidate allowed
            else:
                # Try all valid x
                pass
                
        return None

    # Start search from the beginning, but constrain the early steps to greedy.
    # This is effectively a depth-first search with a heuristic.
    
    # Actually, the logic above is a bit messy. 
    # Let's just try to find ANY solution of length greedy_len + 1.
    # We start with empty state.
    # We use the greedy values as a priority order for candidates.
    
    def dfs(current_values, current_sums, current_prefix):
        if time.time() > deadline:
            return None
        if len(current_values) == target_len:
            return current_values[:]
            
        last_val = current_values[-1] if current_values else 0
        last_prefix = current_prefix[-1]
        
        # Generate candidates
        # Prioritize the greedy choice if available
        candidates = []
        if len(current_values) < greedy_len:
            g_val = greedy_values[len(current_values)]
            if g_val > last_val and g_val <= 256:
                candidates.append(g_val)
                
        # Add other candidates
        for x in range(last_val + 1, 257):
            if x not in candidates:
                candidates.append(x)
                
        # Sort: greedy first, then ascending
        # Actually, just iterate.
        
        for x in candidates:
            P_new = last_prefix + x
            new_diffs = [P_new - p for p in current_prefix]
            if current_sums.intersection(new_diffs):
                continue
                
            new_values = current_values + [x]
            new_sums = current_sums.union(new_diffs)
            new_prefix = current_prefix + [P_new]
            
            res = dfs(new_values, new_sums, new_prefix)
            if res:
                return res
                
        return None

    # Run DFS
    # To prevent explosion, we limit the search.
    # If greedy_len is large, DFS might be too slow.
    # We will only run this if greedy_len is small enough?
    # No, we will run it with a time limit.
    
    better_seq = dfs([], set(), [0])
    
    if better_seq and len(better_seq) > greedy_len:
        return better_seq
    else:
        return greedy_values
