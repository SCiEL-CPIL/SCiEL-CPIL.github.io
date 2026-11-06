import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    max_val = 284
    target_size = 20
    
    # State variables
    best_set = list(prescribed)
    best_size = len(prescribed)
    
    def get_sums(s):
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def can_add(x, current_set, current_sums):
        # Check if x can be added to current_set
        for y in current_set:
            if x + y in current_sums:
                return False
        if 2 * x in current_sums:
            return False
        return True

    def try_greedy(order):
        nonlocal best_set, best_size
        s = list(prescribed)
        sums = get_sums(s)
        for x in order:
            if x in s:
                continue
            if can_add(x, s, sums):
                s.append(x)
                # Update sums
                for y in s:
                    if y != x:
                        sums.add(x + y)
                sums.add(2 * x)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            return True
        return False

    # Phase 1: Greedy Initialization
    remaining = [x for x in range(1, max_val + 1) if x not in prescribed_set]
    
    # Ascending
    try_greedy(remaining)
    # Descending
    try_greedy(list(reversed(remaining)))
    
    # Random Greedy
    rng = random.Random(42)
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        order = remaining[:]
        rng.shuffle(order)
        try_greedy(order)
        
    # If we already found the target, return
    if best_size >= target_size:
        return sorted(best_set)

    # Phase 2: Backtracking with Randomized Restarts
    # We use a global flag or return value to signal success
    
    def dfs(current_set, current_sums, start_val, candidates, time_limit):
        nonlocal best_set, best_size
        
        if time.time() > time_limit:
            return False
            
        current_size = len(current_set)
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size == target_size:
            return True
            
        # Pruning: If we can't reach target_size with remaining candidates
        # candidates is the list of available candidates from start_val onwards
        # Note: In this implementation, we pass a slice or index to avoid copying
        
        # Heuristic: If remaining candidates count < needed, prune
        # But candidates list might have been filtered? 
        # In this simple version, candidates is just the range [start_val, 284].
        # We iterate through them.
        
        needed = target_size - current_size
        # A rough upper bound on how many more we can add is the number of remaining candidates
        # But many will be invalid.
        
        for i in range(len(candidates)):
            x = candidates[i]
            
            if time.time() > time_limit:
                return False
                
            if can_add(x, current_set, current_sums):
                # Add x
                new_set = current_set + [x]
                new_sums = current_sums.copy()
                for y in current_set:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                
                # Recurse with remaining candidates
                # Optimization: We can pass candidates[i+1:] 
                # To avoid slicing cost, we can pass index, but slicing is O(N) and N is small (<=284)
                # However, doing it at every node is expensive.
                # Let's stick to passing the slice for simplicity, as N is small.
                
                if dfs(new_set, new_sums, x + 1, candidates[i+1:], time_limit):
                    return True
                    
        return False

    # We will run multiple DFS attempts with different candidate orderings
    # First attempt: Sorted order (deterministic)
    # Subsequent attempts: Shuffled order
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Time slice for each DFS attempt
        time_slice = 5.0 if attempts == 1 else 3.0
        attempt_deadline = time.time() + time_slice
        
        candidates = list(range(1, max_val + 1))
        # Remove prescribed elements from candidates? 
        # The DFS starts from prescribed, so we should only consider adding others.
        # But the `can_add` and loop handles it. 
        # It's more efficient to filter candidates to those not in prescribed.
        candidates = [x for x in candidates if x not in prescribed_set]
        
        if attempts > 1:
            rng.shuffle(candidates)
            
        # Initial set is prescribed
        initial_set = list(prescribed)
        initial_sums = get_sums(initial_set)
        
        # We need to ensure the candidates list is consistent with the search.
        # The DFS function above assumes `candidates` is a list of values to try.
        # It iterates through them.
        
        if dfs(initial_set, initial_sums, 1, candidates, attempt_deadline):
            if best_size >= target_size:
                return sorted(best_set)
                
    # Phase 3: Local Search Polish
    # If we haven't reached target size, try to improve best_set
    if time.time() < deadline - 10 and best_size < target_size:
        s = best_set[:]
        sums = get_sums(s)
        all_vals = [x for x in range(1, max_val + 1) if x not in s]
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try to add a random element
            random.shuffle(all_vals)
            for x in all_vals:
                if can_add(x, s, sums):
                    s.append(x)
                    for y in s:
                        if y != x:
                            sums.add(x + y)
                    sums.add(2 * x)
                    if len(s) > best_size:
                        best_size = len(s)
                        best_set = s[:]
                    improved = True
                    break
            
            if not improved:
                # Try a swap: remove one non-prescribed, try to add two? 
                # Or just remove one and try to add a different one.
                removable = [x for x in s if x not in prescribed_set]
                if removable:
                    random.shuffle(removable)
                    for x_rm in removable:
                        # Try removing x_rm and then adding a new element
                        s_backup = s[:]
                        sums_backup = sums.copy()
                        
                        # Remove x_rm
                        s.remove(x_rm)
                        # Recompute sums (expensive but N is small)
                        sums = get_sums(s)
                        
                        # Try to add a new element
                        # We can try all candidates or random ones
                        candidates_to_add = [x for x in range(1, max_val + 1) if x not in s]
                        random.shuffle(candidates_to_add)
                        
                        found_add = False
                        for x_add in candidates_to_add:
                            if can_add(x_add, s, sums):
                                s.append(x_add)
                                for y in s:
                                    if y != x_add:
                                        sums.add(x_add + y)
                                sums.add(2 * x_add)
                                found_add = True
                                break
                                
                        if found_add:
                            if len(s) > best_size:
                                best_size = len(s)
                                best_set = s[:]
                                improved = True
                        else:
                            # Restore
                            s = s_backup
                            sums = sums_backup
                            
                        if improved:
                            break

    return sorted(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
    # Verify
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for j in range(i, len(result)):
            s = x + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    print(f"Valid: {valid}")
