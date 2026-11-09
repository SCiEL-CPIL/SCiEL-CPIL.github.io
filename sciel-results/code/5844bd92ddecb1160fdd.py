import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety within 120s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = set(prescribed)
    
    # Helper to calculate initial conflict state
    # Returns (is_valid, conflict_count, sum_counts)
    def analyze_sidon(candidates):
        n = len(candidates)
        if n != len(set(candidates)):
            return False, 100000, {}
        sum_counts = {}
        conflict = 0
        for i in range(n):
            for j in range(i, n):
                s = candidates[i] + candidates[j]
                if s in sum_counts:
                    sum_counts[s] += 1
                    if sum_counts[s] == 2:
                        conflict += 1
                    # If count > 2, conflict doesn't increase further per pair, 
                    # but we track count to know how many "bad" pairs exist?
                    # Actually, for SA, we want to minimize the number of duplicate sums.
                    # A simpler metric: Number of pairs (i,j) such that sum[i,j] == sum[k,l] for some other pair.
                    # Or simply: Count of sums that appear >= 2 times.
                    # Let's stick to: Number of sums that are not unique.
                else:
                    sum_counts[s] = 1
        
        # Recalculate conflict as number of sums with count > 1
        conflict = sum(1 for c in sum_counts.values() if c > 1)
        return conflict == 0, conflict, sum_counts

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
    
    if best_size == 20:
        return best_set

    # 2. Incremental Simulated Annealing
    # State: current_set, current_conflict, sum_counts
    # Move: Change one element. Update sum_counts and conflict incrementally.
    
    def try_find_solution_seeded(target_size, time_limit, seed_set):
        end_time = time.time() + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        
        max_restarts = 10
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            # Initialize state
            if len(seed_set) >= target_size:
                # Ensure we include prescribed elements if we sample
                current_set = random.sample(seed_set, target_size)
                for p in prescribed:
                    if p not in current_set:
                        non_fixed = [i for i, x in enumerate(current_set) if x not in fixed_elements]
                        if non_fixed:
                            idx = random.choice(non_fixed)
                            current_set[idx] = p
            else:
                current_set = list(seed_set)
                needed = target_size - len(current_set)
                pool = [x for x in range(1, 285) if x not in fixed_elements and x not in set(seed_set)]
                random.shuffle(pool)
                current_set.extend(pool[:needed])
            
            random.shuffle(current_set)
            
            # Check for duplicates in current_set itself (shouldn't happen with valid seed, but safe)
            if len(current_set) != len(set(current_set)):
                # Fix duplicates
                seen = set()
                new_set = []
                for x in current_set:
                    if x not in seen:
                        seen.add(x)
                        new_set.append(x)
                current_set = new_set
                # If we lost elements, fill with randoms
                while len(current_set) < target_size:
                    current_set.append(random.randint(1, 284))
            
            is_valid, current_conflict, sum_counts = analyze_sidon(current_set)
            
            if is_valid:
                return current_set
            
            temp = 5.0
            min_temp = 0.01
            cooling_rate = 0.999
            max_steps = 50000
            stagnation_limit = 2000
            stagnation_counter = 0
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                if temp < min_temp:
                    break
                
                # Pick a random index to change
                idx = random.randint(0, len(current_set) - 1)
                
                if current_set[idx] in fixed_elements:
                    continue
                    
                old_val = current_set[idx]
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                
                # Check if new_val is already in the set (causing duplicate element)
                if new_val in current_set:
                    # This move would create a duplicate element, which is invalid.
                    # We can skip or treat as high cost. Skipping is faster.
                    continue

                # Calculate Delta Conflict
                # Removing old_val:
                # For each other element y in current_set (y != old_val):
                #   sum_old = old_val + y
                #   sum_counts[sum_old] decreases by 1
                #   if sum_counts[sum_old] becomes 1, conflict decreases by 1
                #   if sum_counts[sum_old] becomes 0, remove from dict (or leave, doesn't matter much for count)
                # Adding new_val:
                # For each other element y in current_set (y != old_val):
                #   sum_new = new_val + y
                #   sum_counts[sum_new] increases by 1
                #   if sum_counts[sum_new] becomes 2, conflict increases by 1
                
                delta_conflict = 0
                
                # Pre-compute list of other elements to avoid index issues
                others = [current_set[i] for i in range(len(current_set)) if i != idx]
                
                # Remove old_val contributions
                for y in others:
                    s = old_val + y
                    c = sum_counts.get(s, 0)
                    if c > 0:
                        if c == 2:
                            delta_conflict -= 1 # Conflict resolved
                        sum_counts[s] = c - 1
                        if sum_counts[s] == 0:
                            del sum_counts[s]
                
                # Add new_val contributions
                for y in others:
                    s = new_val + y
                    c = sum_counts.get(s, 0)
                    if c == 0:
                        sum_counts[s] = 1
                    else:
                        if c == 1:
                            delta_conflict += 1 # New conflict created
                        sum_counts[s] = c + 1
                
                new_conflict = current_conflict + delta_conflict
                
                # Acceptance probability
                if new_conflict <= current_conflict:
                    current_set[idx] = new_val
                    current_conflict = new_conflict
                    if new_conflict == 0:
                        return current_set
                    stagnation_counter = 0
                else:
                    # Accept with probability exp(-delta/T)
                    try:
                        prob = math.exp(-delta_conflict / temp)
                    except OverflowError:
                        prob = 0.0
                    
                    if random.random() < prob:
                        current_set[idx] = new_val
                        current_conflict = new_conflict
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                
                temp *= cooling_rate
                
                # If stagnating, perform a kick (perturbation)
                if stagnation_counter > stagnation_limit:
                    # Perturb 1-2 random non-fixed elements
                    num_kicks = random.randint(1, 2)
                    for _ in range(num_kicks):
                        k_idx = random.randint(0, len(current_set) - 1)
                        if current_set[k_idx] not in fixed_elements:
                            old_k = current_set[k_idx]
                            new_k = random.randint(1, 284)
                            if new_k not in current_set:
                                # Update state for kick
                                others_k = [current_set[i] for i in range(len(current_set)) if i != k_idx]
                                # Remove old
                                for y in others_k:
                                    s = old_k + y
                                    c = sum_counts.get(s, 0)
                                    if c > 0:
                                        if c == 2: delta_conflict -= 1
                                        sum_counts[s] = c - 1
                                        if sum_counts[s] == 0: del sum_counts[s]
                                # Add new
                                for y in others_k:
                                    s = new_k + y
                                    c = sum_counts.get(s, 0)
                                    if c == 0:
                                        sum_counts[s] = 1
                                    else:
                                        if c == 1: delta_conflict += 1
                                        sum_counts[s] = c + 1
                                current_set[k_idx] = new_k
                                current_conflict += delta_conflict
                                delta_conflict = 0 # Reset for next kick if any
                    stagnation_counter = 0
                    temp = max(temp, 2.0) # Reheat slightly

        return None

    # Try to find larger valid sets, starting from the highest target
    # We have ~115s total. Allocate time wisely.
    # Try 20 first with most time, then 19, 18...
    
    targets = [20, 19, 18, 17, 16, 15]
    time_allocs = [60, 30, 10, 5, 3, 2]
    
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
