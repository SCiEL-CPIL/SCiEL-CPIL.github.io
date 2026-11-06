import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    N = 284
    MAX_SUM = 2 * N

    # Use bytearray for fast sum checking
    # Index 0..MAX_SUM
    initial_used = bytearray(MAX_SUM + 1)
    
    def compute_initial_sums():
        for i in range(len(prescribed)):
            for j in range(i, len(prescribed)):
                s = prescribed[i] + prescribed[j]
                initial_used[s] = 1
        return initial_used.copy()

    # Pre-filter candidates that are compatible with prescribed set
    all_candidates = []
    for x in range(1, N + 1):
        if x in prescribed_set:
            continue
        # Check if x can be added to prescribed set
        valid = True
        if initial_used[2 * x]:
            valid = False
        else:
            for y in prescribed:
                if initial_used[x + y]:
                    valid = False
                    break
        if valid:
            all_candidates.append(x)
    
    best_size = 0
    best_set = list(prescribed)

    def get_sums_set(lst):
        """Return a bytearray of used sums for a given list of elements."""
        used = bytearray(MAX_SUM + 1)
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                used[lst[i] + lst[j]] = 1
        return used

    def is_valid(x, current_set, used):
        """Check if x can be added to current_set given used sums."""
        if used[2 * x]:
            return False
        for y in current_set:
            if used[x + y]:
                return False
        return True

    def greedy_fill(current_set, used, candidates, time_limit=0.5):
        """Greedily add elements to maximize size."""
        start = time.time()
        cands = list(candidates)
        random.shuffle(cands)
        
        # Sort by "safety" - elements that block fewer other candidates?
        # Approximation: just try in random order, but re-shuffle if stuck?
        # For speed, just iterate.
        
        for x in cands:
            if time.time() - start > time_limit:
                break
            if x in current_set:
                continue
            if is_valid(x, current_set, used):
                current_set.add(x)
                used[2 * x] = 1
                for y in current_set:
                    if y != x:
                        used[x + y] = 1
        return current_set, used

    def backtrack_search(current_set, used, candidates, target_size, time_limit=2.0):
        """
        Try to extend current_set to target_size using backtracking.
        Returns (True, new_set) if successful, else (False, best_found_set).
        """
        start = time.time()
        best_found = set(current_set)
        
        # Sort candidates to try "good" ones first
        # Heuristic: candidates that are far from existing elements might be better?
        # Or just randomize and rely on restarts.
        # Let's sort by value to ensure determinism in some runs, or random.
        # Random is better for diversity.
        
        def dfs(sel, used_arr, cand_list, depth):
            nonlocal best_found
            if time.time() - start > time_limit:
                return False
            
            if len(sel) >= target_size:
                best_found = set(sel)
                return True
            
            if len(cand_list) < target_size - len(sel):
                return False

            # Heuristic: Try candidates that are "compatible" with the most other candidates?
            # Too expensive.
            # Try candidates in random order.
            # To improve, we can score candidates by how many sums they *would* block?
            
            # Optimization: Shuffle candidates once per level? No, that's expensive.
            # Just iterate.
            
            # We can prune: if the number of remaining candidates is small, check all.
            
            for i, x in enumerate(cand_list):
                if time.time() - start > time_limit:
                    return False
                
                if x in sel:
                    continue
                
                if not is_valid(x, sel, used_arr):
                    continue
                
                # Add x
                sel.add(x)
                used_arr[2 * x] = 1
                for y in sel:
                    if y != x:
                        used_arr[x + y] = 1
                
                # Recurse
                # Create a new list of candidates excluding x
                # To avoid copying the whole list, we can pass the list and index?
                # But we need to filter out invalid ones for the next level?
                # No, we check validity in the next level's loop.
                # However, passing the same list means we re-check invalid elements.
                # Given the small depth, this is acceptable.
                
                if dfs(sel, used_arr, cand_list, depth + 1):
                    return True
                
                # Undo
                sel.remove(x)
                used_arr[2 * x] = 0
                for y in sel:
                    used_arr[x + y] = 0

            return False

        # Prepare candidates for DFS
        # Exclude elements already in current_set
        dfs_candidates = [x for x in candidates if x not in current_set]
        random.shuffle(dfs_candidates)
        
        # Run DFS
        dfs(set(current_set), used, dfs_candidates, 0)
        
        return best_found

    # Main Loop
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # 1. Initialize
        current_set = set(prescribed_set)
        current_used = compute_initial_sums()
        
        # 2. Greedy Fill
        # Use a time slice for greedy
        greedy_time = 0.5
        current_set, current_used = greedy_fill(current_set, current_used, all_candidates, greedy_time)
        
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            if best_size >= 20:
                break

        # 3. Try to reach 20 (or best_size + 1)
        target = 20
        if best_size >= 19:
            target = 20
        
        # If greedy got us close, try backtracking
        if current_size < target:
            # Determine time for backtracking
            remaining = deadline - time.time()
            bt_time = min(2.0, max(0.5, remaining * 0.05))
            
            # We pass the current set and used sums
            # Note: backtrack_search modifies the set it's passed if we're not careful, 
            # but we pass a copy inside the function? 
            # In my implementation above, `dfs` takes `sel` and `used_arr` by reference 
            # and undoes changes. So it's safe if we pass copies or handle it carefully.
            # `current_set` is a set. `current_used` is a bytearray.
            # We should pass copies to `backtrack_search` to not corrupt the state if we want to retry.
            
            # Actually, `backtrack_search` creates `best_found` and returns it.
            # It uses `dfs` which modifies local `sel` and `used_arr` (passed by ref) but undoes them.
            # So we can pass `current_set` and `current_used` directly, as long as 
            # `dfs` restores them.
            
            new_set = backtrack_search(current_set, current_used, all_candidates, target, bt_time)
            
            if len(new_set) > current_size:
                current_set = new_set
                current_used = get_sums_set(list(current_set))
                current_size = len(current_set)
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    if best_size >= 20:
                        break

        # 4. Kick
        if current_size < 20:
            # Remove 1 or 2 random elements (not prescribed)
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                kick_size = 1 if random.random() < 0.7 else 2
                if len(removable) >= kick_size:
                    to_remove = random.sample(removable, kick_size)
                    for x in to_remove:
                        current_set.remove(x)
                    # Recalculate used sums
                    current_used = get_sums_set(list(current_set))
                    
                    # Try to refill
                    # Greedy fill again
                    current_set, current_used = greedy_fill(current_set, current_used, all_candidates, 0.5)
                    current_size = len(current_set)
                    
                    if current_size < 20:
                        # Backtrack again
                        remaining = deadline - time.time()
                        if remaining > 1:
                            bt_time = min(1.5, remaining * 0.05)
                            new_set = backtrack_search(current_set, current_used, all_candidates, 20, bt_time)
                            if len(new_set) > current_size:
                                current_set = new_set
                                current_used = get_sums_set(list(current_set))
                                current_size = len(current_set)
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = list(current_set)
                                    if best_size >= 20:
                                        break

    # Final Polish: Single Swap
    if time.time() < deadline - 2:
        best_sums = get_sums_set(best_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            removable = [x for x in best_set if x not in prescribed_set]
            for x_remove in removable:
                if time.time() >= deadline - 2:
                    break
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums_set(list(temp_set))
                
                # Try to add any valid candidate
                # Sort candidates by "potential"?
                # Just iterate all
                for y_add in all_candidates:
                    if y_add in temp_set:
                        continue
                    if is_valid(y_add, temp_set, temp_sums):
                        # If we can add y_add, size increases by 1 (since we removed 1)
                        # Wait, if we remove 1 and add 1, size is same.
                        # We want to increase size.
                        # So we should only do this if we can add WITHOUT removing?
                        # The "Final Polish" in the original code tried to add to the set 
                        # after removing, but that keeps size same unless we add multiple?
                        # No, the original code checked `if len(new_set) > len(best_set)`.
                        # If we remove 1 and add 1, len is same.
                        # So this polish only helps if we can find a configuration where 
                        # removing x allows adding y AND z? No, single step.
                        
                        # Actually, if we are at local optimum of size K, 
                        # removing 1 element allows adding 2 elements?
                        # That's a 1-for-2 swap.
                        # My loop above only adds 1.
                        
                        # Let's modify: Try to add 1 element to the set WITHOUT removing.
                        # If possible, we increase size.
                        pass # Handled in main loop greedy/backtrack.
                        
                # To do a 1-for-2 swap:
                # Remove x.
                # Try to add y.
                # If valid, try to add z to {temp_set, y}.
                # If valid, we have size K+1.
                
                # Let's implement 1-for-2
                found_pair = False
                for y_add in all_candidates:
                    if y_add in temp_set:
                        continue
                    if is_valid(y_add, temp_set, temp_sums):
                        # Tentatively add y_add
                        temp_set.add(y_add)
                        temp_sums[2 * y_add] = 1
                        for y in temp_set:
                            if y != y_add:
                                temp_sums[y_add + y] = 1
                                
                        # Now try to add z_add
                        for z_add in all_candidates:
                            if z_add in temp_set:
                                continue
                            if is_valid(z_add, temp_set, temp_sums):
                                # Success!
                                best_set = list(temp_set)
                                best_size = len(best_set)
                                improved = True
                                found_pair = True
                                break
                            # Undo z? No, we break out.
                        
                        if found_pair:
                            break
                            
                        # Undo y_add
                        temp_set.remove(y_add)
                        temp_sums[2 * y_add] = 0
                        for y in temp_set:
                            temp_sums[y_add + y] = 0
                        
                        if found_pair:
                            break
                            
                if found_pair:
                    break

    # Validation
    final_sums = get_sums_set(best_set)
    expected_sums = len(best_set) * (len(best_set) + 1) // 2
    actual_sums = sum(final_sums)
    if actual_sums != expected_sums:
        return sorted(prescribed)

    return sorted(best_set)
