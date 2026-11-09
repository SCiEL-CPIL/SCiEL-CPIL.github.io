import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    prescribed = [1, 2, 5, 11]
    pres_set = set(prescribed)
    N = 284
    target_size = 20
    
    # Initial state
    current_set = sorted(prescribed)
    used_sums = set()
    for i in range(len(current_set)):
        for j in range(i, len(current_set)):
            used_sums.add(current_set[i] + current_set[j])
            
    # Candidates are numbers 1..284 not in current_set
    # We will use backtracking to add elements
    
    def is_valid_addition(x, current_set, used_sums):
        # Check if adding x creates any duplicate sums
        # Sums involving x: x+x and x+y for all y in current_set
        # Note: We assume x is not in current_set and x is not in used_sums as a value, 
        # but we only care about sum collisions.
        
        # Check x+x
        if x + x in used_sums:
            return False
        
        # Check x+y for y in current_set
        for y in current_set:
            if x + y in used_sums:
                return False
        return True

    def get_new_sums(x, current_set):
        sums = {x + x}
        for y in current_set:
            sums.add(x + y)
        return sums

    # Backtracking search
    # We want to find a set of size 20.
    # We'll try to add elements one by one.
    # To improve performance, we can sort candidates.
    # A good heuristic: Try candidates that are "less restrictive" or just iterate 1..N.
    # Given N=284 and depth=16 (20-4), simple iteration might be fast enough with pruning.
    
    best_found = list(current_set)
    
    def backtrack(depth, max_depth, current_set, used_sums):
        nonlocal best_found
        
        if time.time() > deadline:
            return False
            
        if len(current_set) == max_depth:
            best_found = list(current_set)
            return True
            
        if len(current_set) > len(best_found):
            best_found = list(current_set)
            
        # Generate candidates
        # To speed up, we can pre-compute or just iterate.
        # Iterating 1..N is O(N) per node.
        # Depth is ~16. Total nodes could be large, but pruning is strong.
        
        # Heuristic: Try to add numbers that are "spread out"?
        # Actually, for Sidon sets, smaller numbers tend to be more constrained 
        # because they create smaller sums which might collide with other small sums.
        # However, large numbers create large sums which are less likely to collide 
        # with sums of smaller numbers? 
        # Let's just iterate from 1 to N.
        
        for x in range(1, N + 1):
            if x in current_set:
                continue
                
            if is_valid_addition(x, current_set, used_sums):
                # Add x
                new_sums = get_new_sums(x, current_set)
                current_set.append(x)
                used_sums.update(new_sums)
                
                # Recurse
                if backtrack(depth + 1, max_depth, current_set, used_sums):
                    return True
                    
                # Backtrack
                current_set.pop()
                used_sums.difference_update(new_sums)
                
        return False

    # Try to reach size 20
    # We start with 4 elements. We need to add 16 more.
    # We call backtrack with max_depth = 20.
    
    # To make it faster, we can try multiple strategies or just rely on the backtracker.
    # The backtracker as written explores the search space.
    # To ensure we don't miss a solution due to ordering, we might need to be careful.
    # But for this problem, a solution of size 20 exists.
    
    # Optimization: If the simple backtracker is too slow, we might need better heuristics.
    # Let's try to run it.
    
    # We use a list for current_set to allow in-place modification for speed.
    # used_sums is a set.
    
    # One issue: `x in current_set` is O(N). We should use a set for membership.
    # Let's refactor to use a set for current_set membership and a list for iteration order.
    
    current_set_list = list(current_set)
    current_set_set = set(current_set)
    
    def backtrack_opt(depth, max_depth, current_set_list, current_set_set, used_sums):
        nonlocal best_found
        
        if time.time() > deadline:
            return False
            
        if len(current_set_list) == max_depth:
            best_found = list(current_set_list)
            return True
            
        if len(current_set_list) > len(best_found):
            best_found = list(current_set_list)
            
        for x in range(1, N + 1):
            if x in current_set_set:
                continue
                
            if is_valid_addition(x, current_set_list, used_sums):
                # Add x
                new_sums = get_new_sums(x, current_set_list)
                current_set_list.append(x)
                current_set_set.add(x)
                used_sums.update(new_sums)
                
                # Recurse
                if backtrack_opt(depth + 1, max_depth, current_set_list, current_set_set, used_sums):
                    return True
                    
                # Backtrack
                current_set_list.pop()
                current_set_set.remove(x)
                used_sums.difference_update(new_sums)
                
        return False

    # Start search for size 20
    # If it fails, we might have a best_found of size < 20.
    # We can try to improve best_found if it's not 20, but 20 is the max.
    
    # To increase the chance of finding 20 quickly, we can randomize the order of candidates 
    # if the first deterministic attempt fails or takes too long?
    # But deterministic is usually faster if it finds a solution.
    # Let's try deterministic first.
    
    backtrack_opt(0, 20, current_set_list, current_set_set, used_sums)
    
    # If we didn't find 20, we return the best found.
    # However, the problem states "Maximize the size".
    # If 20 is not found, maybe 19 is?
    # The backtracker already updates best_found.
    
    # Final validation
    # Ensure best_found is valid
    if not is_valid_set(best_found, pres_set, N):
        # Fallback to greedy if something went wrong
        best_found = greedy_fallback(prescribed, N)
        
    return best_found

def is_valid_set(s, pres_set, N):
    if len(s) != len(set(s)):
        return False
    if not pres_set.issubset(set(s)):
        return False
    for v in s:
        if v < 1 or v > N:
            return False
    seen = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            v = s[i] + s[j]
            if v in seen:
                return False
            seen.add(v)
    return True

def greedy_fallback(prescribed, N):
    arr = list(prescribed)
    used = set(arr)
    sums = set()
    for a in arr:
        for b in arr:
            if a <= b:
                sums.add(a + b)
    for x in range(1, N + 1):
        if x in used:
            continue
        ok = True
        new_sums = []
        for a in arr:
            s = a + x
            if s in sums:
                ok = False
                break
            new_sums.append(s)
        if ok and (x + x) in sums:
            ok = False
        if ok:
            sums.update(new_sums)
            sums.add(x + x)
            used.add(x)
            arr.append(x)
    return arr
