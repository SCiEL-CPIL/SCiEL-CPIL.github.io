def construct():
    start_time = time.time()
    deadline = start_time + 145

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    n = max_val

    def get_sums(s: Set[int]) -> Set[int]:
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def can_add(x: int, s: Set[int], used_sums: Set[int]) -> bool:
        # Check if x+x and x+y conflict
        if 2 * x in used_sums:
            return False
        for y in s:
            if x + y in used_sums:
                return False
        return True

    def get_new_sums(x: int, s: Set[int]):
        return {2 * x} | {x + y for y in s}

    # --- Phase 1: Greedy Initialization ---
    def greedy_init(order: List[int] = None) -> List[int]:
        if order is None:
            order = list(range(1, n + 1))
        
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        
        for x in order:
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                current_set.add(x)
                new_sums = get_new_sums(x, current_set)
                # Note: get_new_sums uses current_set which now includes x? 
                # No, standard practice: calculate sums involving x and previous elements.
                # My get_new_sums(x, s) where s is the set BEFORE adding x is safer for logic, 
                # but here I added x first. Let's fix can_add/get_new_sums usage.
                
                # Correction:
                # can_add checks against used_sums (sums of old set).
                # If valid, we add x.
                # New sums are {x+x} U {x+y for y in old_set}.
                # So we should calculate sums based on the set BEFORE adding x?
                # Or just calculate {x+y for y in current_set} where current_set includes x?
                # If current_set includes x, {x+y} includes 2x. So {x+y for y in current_set} is correct.
                
                # Let's re-verify:
                # old_set = S
                # new_set = S U {x}
                # new_sums = {a+b | a,b in new_set, a<=b}
                # old_sums = {a+b | a,b in S, a<=b}
                # added_sums = {x+x} U {x+y | y in S}
                # So if I do:
                # current_set.add(x)
                # new_sums = {x + y for y in current_set}
                # This includes 2x and x+y for all y in S (since x is in current_set).
                # This is correct.
                
                used_sums.update(get_new_sums(x, current_set)) # This function is slightly misnamed if it expects old set, but logic holds if it uses current.
                # Actually, let's just update used_sums directly.
        
        return sorted(current_set)

    # The previous get_new_sums logic in the draft was:
    # def get_new_sums(x, s): return {2*x} | {x+y for y in s}
    # If s is the set BEFORE adding x, this is correct.
    # If s is the set AFTER adding x, {x+y for y in s} includes 2x.
    # Let's stick to: calculate sums involving x with elements in the set BEFORE adding x, plus 2x.
    
    def try_greedy_backtrack(target_size: int, rng: random.Random) -> List[int]:
        """
        Attempts to find a valid set of size target_size using randomized greedy with backtracking.
        """
        # Candidates are all non-prescribed numbers
        candidates = [x for x in range(1, n + 1) if x not in prescribed_set]
        rng.shuffle(candidates)
        
        # We want to pick target_size - len(prescribed) elements from candidates.
        # We can use a simple DFS with pruning.
        # State: (current_set, used_sums, candidate_index)
        
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        needed = target_size - len(current_set)
        
        if needed <= 0:
            return sorted(current_set)
        
        # DFS
        def dfs(idx: int, remaining_needed: int):
            if remaining_needed == 0:
                return True
            if idx >= len(candidates):
                return False
            
            # Pruning: not enough candidates left
            if len(candidates) - idx < remaining_needed:
                return False
            
            for i in range(idx, len(candidates)):
                x = candidates[i]
                if can_add(x, current_set, used_sums):
                    # Add x
                    current_set.add(x)
                    new_sums = {2*x} | {x + y for y in current_set if y != x}
                    # Wait, if I add x to current_set, then {x+y for y in current_set} includes 2x.
                    # So new_sums = {x + y for y in current_set}
                    added_sums = {x + y for y in current_set}
                    
                    # Store old sums to rollback? 
                    # It's better to pass used_sums as a parameter or copy it.
                    # Copying is O(N^2). N is small (~20). So copying is fine.
                    
                    # Actually, let's just recurse with the updated state.
                    # We need to rollback.
                    # Since we are modifying current_set in place, we must be careful.
                    
                    # Let's use a cleaner recursive structure that doesn't mutate global state in a risky way,
                    # or carefully rollback.
                    
                    # Rollback:
                    # Remove x from current_set
                    # Remove added_sums from used_sums? 
                    # used_sums is a set. We can't easily "remove" if a sum was already there?
                    # No, Sidon property ensures all sums are distinct. So added_sums are new.
                    # So we can just discard the new used_sums state and restore old one?
                    # Yes, passing used_sums by value (copy) is safest and fast enough for N=20.
                    
                    # Let's refactor dfs to take state as arguments.
                    pass
        
        # Refactored DFS
        def dfs_rec(current_set, used_sums, start_idx, remaining_needed):
            if remaining_needed == 0:
                return current_set
            
            # Pruning
            if len(candidates) - start_idx < remaining_needed:
                return None
            
            for i in range(start_idx, len(candidates)):
                x = candidates[i]
                if can_add(x, current_set, used_sums):
                    new_set = current_set | {x}
                    new_sums = used_sums | {x + y for y in current_set} | {2*x}
                    
                    # Check time
                    if time.time() > deadline:
                        return None
                    
                    res = dfs_rec(new_set, new_sums, i + 1, remaining_needed - 1)
                    if res is not None:
                        return res
            return None

        result = dfs_rec(set(prescribed), get_sums(set(prescribed)), 0, needed)
        if result:
            return sorted(result)
        return None

    best_set = greedy_init()
    best_size = len(best_set)
    
    # Phase 2: Iterated Local Search / Randomized Backtracking
    import random
    
    # Try to find larger sets
    target = best_size + 1
    max_target = 25 # Safe upper bound, theoretically 20
    
    # We will run multiple trials.
    # Each trial:
    # 1. Try to find a set of size 'target' using randomized backtracking.
    # 2. If found, update best_size and target.
    # 3. If not found, try to improve current best via local search (swaps).
    
    rng = random.Random(42)
    
    # Local Search on best_set
    def local_search_improve(s: List[int]) -> List[int]:
        current_set = set(s)
        current_sums = get_sums(current_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if can_add(x, current_set, current_sums):
                    current_set.add(x)
                    current_sums.update({x + y for y in current_set})
                    improved = True
                    break # Restart loop to check new state
            
            if not improved:
                # Try swap
                removable = [x for x in current_set if x not in prescribed_set]
                if not removable:
                    break
                # Randomly pick a few swaps to try
                for _ in range(10):
                    x_remove = rng.choice(removable)
                    candidates = [x for x in range(1, n + 1) if x not in current_set]
                    if not candidates:
                        break
                    x_add = rng.choice(candidates)
                    
                    # Check if swap is valid
                    temp_set = set(current_set)
                    temp_set.remove(x_remove)
                    if can_add(x_add, temp_set, get_sums(temp_set)):
                        # Perform swap
                        current_set = temp_set
                        current_set.add(x_add)
                        current_sums = get_sums(current_set)
                        improved = True
                        break
        return sorted(current_set)

    # Main Loop
    while time.time() < deadline - 5:
        # 1. Try to jump to target size via randomized backtracking
        if target <= max_target:
            found = try_greedy_backtrack(target, rng)
            if found:
                best_set = found
                best_size = len(best_set)
                target = best_size + 1
                continue
        
        # 2. Local search on current best
        if time.time() < deadline - 10:
            improved_set = local_search_improve(best_set)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
                target = best_size + 1
        
        # 3. Random restart: Generate a new random greedy set and improve it
        if time.time() < deadline - 20:
            # Create a random order
            order = list(range(1, n + 1))
            rng.shuffle(order)
            # Ensure prescribed are always considered? Greedy handles it.
            new_set = greedy_init(order)
            # Improve it
            new_set = local_search_improve(new_set)
            if len(new_set) > best_size:
                best_set = new_set
                best_size = len(best_set)
                target = best_size + 1

    # Final verification
    final_set = set(best_set)
    sums = get_sums(final_set)
    expected = len(final_set) * (len(final_set) + 1) // 2
    if len(sums) != expected:
        best_set = list(prescribed)

    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")
