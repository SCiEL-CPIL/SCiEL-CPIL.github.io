import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety within 120s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = set(prescribed)
    
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

    # 2. Seeded Simulated Annealing to find larger valid sets
    # Strategy: For each target size, start from greedy base + random additions,
    # then use SA to resolve conflicts
    
    def try_find_solution_seeded(target_size, time_limit, seed_set):
        end_time = time.time() + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        
        max_restarts = 30
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            # Start with seed_set (which is valid)
            # If len(seed_set) > target_size, we can't use it directly as a base for growth?
            # No, we want a set of EXACTLY target_size.
            # If seed_set is smaller, we add randoms.
            if len(seed_set) >= target_size:
                # Take a random subset of seed_set of size target_size
                current_set = random.sample(seed_set, target_size)
                # Ensure prescribed elements are included
                for p in prescribed:
                    if p not in current_set:
                        # Replace a random non-prescribed element
                        non_fixed = [i for i, x in enumerate(current_set) if x not in fixed_elements]
                        if non_fixed:
                            idx = random.choice(non_fixed)
                            current_set[idx] = p
            else:
                # Start with seed_set and add random elements to reach target_size
                current_set = list(seed_set)
                needed = target_size - len(current_set)
                
                pool = [x for x in range(1, 285) if x not in fixed_elements and x not in set(seed_set)]
                random.shuffle(pool)
                current_set.extend(pool[:needed])
            
            random.shuffle(current_set)
            
            # Identify fixed indices
            is_fixed = [x in fixed_elements for x in current_set]
            
            # Evaluate initial state
            is_valid, current_conflict = check_sidon(current_set)
            
            if is_valid:
                return current_set
            
            # Simulated Annealing
            temp = 5.0
            min_temp = 0.05
            cooling_rate = 0.9997
            max_steps = 100000
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                    
                if temp < min_temp:
                    break
                
                # Pick a random index to change
                idx = random.randint(0, len(current_set) - 1)
                
                if is_fixed[idx]:
                    continue
                    
                old_val = current_set[idx]
                # Choose a new random value from the pool
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                
                current_set[idx] = new_val
                
                # Evaluate new state
                is_valid, new_conflict = check_sidon(current_set)
                
                if is_valid:
                    return current_set
                    
                delta = new_conflict - current_conflict
                
                if delta <= 0:
                    current_conflict = new_conflict
                else:
                    # Accept worse solution with probability
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

    # Try to find larger valid sets, starting from the highest target
    # We have ~115s total. Allocate time wisely.
    # Try 20 first with most time, then 19, 18...
    
    targets = [20, 19, 18, 17, 16, 15]
    time_allocs = [50, 30, 15, 10, 5, 3]
    
    for i, target in enumerate(targets):
        if target <= best_size:
            continue
            
        alloc = time_allocs[i]
        # Don't exceed deadline
        if time.time() + alloc > deadline:
            alloc = int(deadline - time.time())
            
        if alloc <= 0:
            break
            
        res = try_find_solution_seeded(target, alloc, best_set)
        
        if res is not None:
            best_set = res
            best_size = len(res)
            break  # Found a valid larger set, stop searching lower targets
    
    return best_set

# Main entry point
if __name__ == "__main__":
    res = construct()
    print(len(res))
