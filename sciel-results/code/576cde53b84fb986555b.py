"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety within 120s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = set(prescribed)
    
    # Precompute: for each possible sum s, which pairs (i,j) with i<=j produce it
    # This lets us do O(1) conflict counting per element change
    # sums range from 1+1=2 to 284+284=568
    MAX_SUM = 568
    
    # Helper to check if a list is a valid Sidon set
    # Returns (is_valid, conflict_count)
    def check_sidon(candidates):
        n = len(candidates)
        if n != len(set(candidates)):
            return False, 100000
        sums = set()
        conflict = 0
        for i in range(n):
            for j in range(i, n):
                s = candidates[i] + candidates[j]
                if s in sums:
                    conflict += 1
                else:
                    sums.add(s)
        return conflict == 0, conflict

    # Fast conflict count using sum -> count mapping
    def fast_conflict_count(arr):
        """Count the number of duplicate sums in arr."""
        n = len(arr)
        if n != len(set(arr)):
            return 100000  # duplicates present
        sum_counts = {}
        conflict = 0
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in sum_counts:
                    conflict += 1
                else:
                    sum_counts[s] = 1
        return conflict

    # 1. Greedy Construction to get a strong baseline
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
            
    best_set = a
    best_size = len(best_set)
    
    # If we already hit the theoretical max (20), return immediately
    if best_size == 20:
        return best_set

    # 2. Improved Simulated Annealing with incremental conflict tracking
    # Key insight: when we change one element, only sums involving that element change.
    # We can maintain a sum_counts dict and update it incrementally.
    
    def try_find_solution_seeded(target_size, time_limit, seed_set):
        nonlocal start_time, deadline
        end_time = start_time + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        
        max_restarts = 30
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            if len(seed_set) > target_size:
                return None
                
            current_set = list(seed_set)
            needed = target_size - len(current_set)
            
            if needed < 0:
                return None
                
            pool = [x for x in range(1, 285) if x not in set(current_set)]
            random.shuffle(pool)
            current_set.extend(pool[:needed])
            random.shuffle(current_set)
            
            # Identify fixed indices (prescribed)
            is_fixed = [x in fixed_elements for x in current_set]
            
            # Build initial sum_counts and conflict count
            n = len(current_set)
            sum_counts = {}
            current_conflict = 0
            for i in range(n):
                for j in range(i, n):
                    s = current_set[i] + current_set[j]
                    if s in sum_counts:
                        current_conflict += 1
                        sum_counts[s] += 1
                    else:
                        sum_counts[s] = 1
            
            if current_conflict == 0:
                return current_set
            
            # Simulated Annealing with incremental updates
            temp = 8.0
            min_temp = 0.01
            cooling_rate = 0.9997
            max_steps = 500000
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                    
                if temp < min_temp:
                    break
                    
                # Pick a random index to change
                idx = random.randint(0, n - 1)
                
                if is_fixed[idx]:
                    continue
                    
                old_val = current_set[idx]
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                
                # Remove old_val's contributions to sum_counts
                # old_val contributes with every other element j != idx
                # and with itself (2*old_val)
                # Sums to remove: old_val + current_set[j] for all j != idx, and 2*old_val
                # We need to decrement counts and adjust conflict count
                # Conflict count changes: for each sum s involving old_val,
                #   if count[s] was 2, removing one makes it 1 -> conflict decreases by 1
                #   if count[s] was >2, removing one -> conflict unchanged (still >=2)
                #   if count[s] was 1, removing one -> count becomes 0, but that sum no longer exists
                
                # Actually, let's think more carefully:
                # conflict = sum over all sums s of max(0, count[s] - 1)
                # When we remove a sum s (decrement count[s]):
                #   if count[s] was 1: conflict unchanged (was 0, still 0)
                #   if count[s] was 2: conflict decreases by 1 (was 1, now 0)
                #   if count[s] was k>2: conflict unchanged (was k-1, still k-1)
                # When we add a sum s (increment count[s]):
                #   if count[s] was 0: conflict unchanged (was 0, still 0)
                #   if count[s] was 1: conflict increases by 1 (was 0, now 1)
                #   if count[s] was k>=2: conflict unchanged (was k-1, still k-1)
                
                # So: delta_conflict when removing = -1 if count[s] was 2, else 0
                #     delta_conflict when adding = +1 if count[s] was 1, else 0
                #     (where "count[s]" is the value BEFORE the operation)
                
                # Remove old_val contributions
                for j in range(n):
                    if j == idx:
                        continue
                    s = old_val + current_set[j]
                    c = sum_counts.get(s, 0)
                    if c == 2:
                        current_conflict -= 1
                    elif c == 1:
                        # count becomes 0, sum disappears
                        pass
                    sum_counts[s] = c - 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
                
                # Remove 2*old_val
                s = 2 * old_val
                c = sum_counts.get(s, 0)
                if c == 2:
                    current_conflict -= 1
                sum_counts[s] = c - 1
                if sum_counts[s] == 0:
                    del sum_counts[s]
                
                # Set new value
                current_set[idx] = new_val
                
                # Check for duplicate values
                has_dup = False
                for j in range(n):
                    if j != idx and current_set[j] == new_val:
                        has_dup = True
                        break
                
                if has_dup:
                    # This is bad, but SA can handle it. Let's just add the contributions
                    # and let the high conflict push us away.
                    pass
                
                # Add new_val contributions
                for j in range(n):
                    if j == idx:
                        continue
                    s = new_val + current_set[j]
                    c = sum_counts.get(s, 0)
                    if c == 1:
                        current_conflict += 1
                    sum_counts[s] = c + 1
                
                # Add 2*new_val
                s = 2 * new_val
                c = sum_counts.get(s, 0)
                if c == 1:
                    current_conflict += 1
                sum_counts[s] = c + 1
                
                if current_conflict == 0:
                    return current_set
                
                # Simulated annealing acceptance
                # We need to compute delta. But we didn't save old_conflict explicitly.
                # Let's restructure: save old_conflict before modifying.
                # Actually, let me redo this more carefully.
                
                # Roll back for now and use a simpler approach: just recompute
                # The incremental approach is getting complex. Let's use a hybrid:
                # For small n (<=25), recompute is fast enough.
                pass

            # If we get here, the SA loop didn't find a solution in this restart
            continue

        return None

    # Simpler and more robust approach: use full recomputation but with optimized inner loop
    def try_find_solution_v2(target_size, time_limit, seed_set):
        nonlocal start_time, deadline
        end_time = start_time + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        n_target = target_size
        
        max_restarts = 40
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            if len(seed_set) > target_size:
                return None
                
            current_set = list(seed_set)
            needed = target_size - len(current_set)
            
            if needed < 0:
                return None
                
            pool = [x for x in range(1, 285) if x not in set(current_set)]
            random.shuffle(pool)
            current_set.extend(pool[:needed])
            random.shuffle(current_set)
            
            # Identify fixed indices (prescribed)
            is_fixed = [x in fixed_elements for x in current_set]
            n = len(current_set)
            
            # Evaluate initial state
            def compute_conflict(arr):
                """Compute conflict count for a set of values."""
                if len(arr) != len(set(arr)):
                    return 100000
                seen = {}
                conf = 0
                for i in range(len(arr)):
                    for j in range(i, len(arr)):
                        s = arr[i] + arr[j]
                        if s in seen:
                            conf += 1
                        else:
                            seen[s] = True
                return conf
            
            current_conflict = compute_conflict(current_set)
            
            if current_conflict == 0:
                return current_set
            
            # Simulated Annealing
            temp = 6.0
            min_temp = 0.01
            cooling_rate = 0.9998
            max_steps = 300000
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                    
                if temp < min_temp:
                    break
                    
                # Pick a random index to change
                idx = random.randint(0, n - 1)
                
                if is_fixed[idx]:
                    continue
                    
                old_val = current_set[idx]
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                
                current_set[idx] = new_val
                new_conflict = compute_conflict(current_set)
                
                delta = new_conflict - current_conflict
                
                if delta <= 0:
                    current_conflict = new_conflict
                    if current_conflict == 0:
                        return current_set
                else:
                    try:
                        prob = math.exp(-delta / temp)
                    except OverflowError:
                        prob = 0.0
                    
                    if random.random() < prob:
                        current_conflict = new_conflict
                    else:
                        current_set[idx] = old_val
                        
                temp *= cooling_rate

        return None

    # Try to reach 20
    if best_size < 20:
        res = try_find_solution_v2(20, 70, best_set)
        if res:
            return res

    # If that failed, try to reach 19 (if we haven't already)
    if best_size < 19:
        res = try_find_solution_v2(19, 25, best_set)
        if res:
            return res

    # If that failed, try to reach 18
    if best_size < 18:
        res = try_find_solution_v2(18, 15, best_set)
        if res:
            return res

    # If nothing improved, return the best we have (the greedy one)
    return best_set

# EVOLVE-BLOCK-END
