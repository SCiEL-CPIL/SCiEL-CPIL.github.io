import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Fixed elements
    fixed = [1, 2, 5, 11]
    
    # Initial sums
    used_sums = [False] * (2 * 284 + 1)
    current_set = list(fixed)
    
    for i in range(len(current_set)):
        for j in range(i, len(current_set)):
            s = current_set[i] + current_set[j]
            used_sums[s] = True
            
    best_size = len(current_set)
    best_set = list(current_set)
    
    # We want to maximize size.
    # Try to add elements to reach size 20, then 19, etc.
    # But we don't know the max. Let's just do a DFS that tries to add as many as possible.
    # Actually, a standard backtracking to find ANY valid set of size K is better.
    # Let's try K from 20 down to 5.
    
    def can_add(x):
        # Check if x can be added
        # New sums: x + y for y in current_set, and 2*x
        for y in current_set:
            s = x + y
            if s > 568: # Max possible sum is 284+284=568
                continue
            if used_sums[s]:
                return False
        if used_sums[2 * x]:
            return False
        return True

    def dfs(target_size, start_idx):
        nonlocal best_size, best_set, used_sums, current_set, deadline
        
        if time.time() > deadline:
            return False
            
        if len(current_set) == target_size:
            best_size = target_size
            best_set = list(current_set)
            return True
            
        # Pruning: if we can't reach target_size with remaining candidates
        # Candidates are from 12 to 284.
        # We are iterating in some order.
        # Let's iterate from 284 down to 12 to spread them out.
        
        # Heuristic: If remaining slots > remaining candidates, fail
        # But we don't track start_idx strictly for all candidates if we re-scan.
        # Better to pass a list of candidates or an index.
        
        # Let's use a global candidate list sorted descending.
        # But we need to ensure we don't reuse candidates.
        # Since we are adding one by one, and checking against current_set,
        # we just need to pick the next x from the pool of unused candidates.
        
        # To allow backtracking, we iterate through candidates.
        
        # Optimization: 
        # If len(current_set) + (284 - 11) < target_size: return False (always true for small target)
        
        for x in range(284, 11, -1):
            if time.time() > deadline:
                return False
                
            # Skip if x is already in current_set
            if x in current_set:
                continue
                
            if can_add(x):
                # Add x
                current_set.append(x)
                
                # Mark sums
                sums_to_mark = []
                for y in current_set[:-1]: # y in old current_set
                    s = x + y
                    used_sums[s] = True
                    sums_to_mark.append(s)
                s = 2 * x
                used_sums[s] = True
                sums_to_mark.append(s)
                
                if dfs(target_size, x):
                    return True
                    
                # Undo
                for s in sums_to_mark:
                    used_sums[s] = False
                current_set.pop()
                
        return False

    # Try sizes from 20 down to 5
    # Note: The prompt says max is 20.
    for target_size in range(20, 4, -1):
        if time.time() > deadline:
            break
        # Reset state for each target size? 
        # No, we can just run dfs. But dfs modifies global state.
        # We need to reset current_set and used_sums for each target size attempt?
        # Actually, if we fail to find size 20, we should try size 19.
        # The search for size 19 is a subset of the search for size 20.
        # But the order of exploration might be different.
        # To be safe, we can just run the solver for the max possible size, and if it fails, reduce.
        # However, a DFS that stops at the first solution of size K is efficient.
        
        # Let's reset the state to the initial fixed set before each target size attempt.
        # This ensures we don't carry over bad paths.
        # But it's expensive to reset.
        # Alternatively, we can just do one DFS that tries to maximize size.
        # But DFS doesn't naturally maximize unless we explore all branches.
        # We want the FIRST solution of size 20. If none, size 19.
        
        # Let's re-initialize state for each target size.
        current_set = list(fixed)
        used_sums = [False] * (2 * 284 + 1)
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                used_sums[current_set[i] + current_set[j]] = True
        
        if dfs(target_size, 0):
            return best_set
            
    return best_set
