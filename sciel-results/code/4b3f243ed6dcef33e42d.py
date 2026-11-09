"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for safety within 120s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = set(prescribed)
    
    # Helper to check if a list is a valid Sidon set
    # Returns (is_valid, conflict_count)
    # We use a dictionary to track sums for O(1) lookup
    def check_sidon(candidates):
        n = len(candidates)
        if n != len(set(candidates)):
            return False, 100000
        
        sums = {}
        conflict = 0
        for i in range(n):
            for j in range(i, n):
                s = candidates[i] + candidates[j]
                if s in sums:
                    conflict += 1
                else:
                    sums[s] = 1
        return conflict == 0, conflict

    # 1. Greedy Construction to get a strong baseline
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    
    # Standard greedy: add smallest possible number
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

    # 2. Local Search / Simulated Annealing to improve
    # Structural Change: Instead of just random restarts or simple SA, we use a "Swap-based" 
    # or "Insertion/Replacement" hybrid. 
    # Key Insight from Inspiration: Randomized greedy restarts + swap-based local search is robust.
    # We will implement a local search that tries to replace one element with another to fix conflicts,
    # and also tries to insert new elements if we are below target size (though we fix size here).
    
    # We want to find a valid set of size TargetSize.
    # We initialize with the prescribed elements and random others, then optimize.
    
    def try_find_solution(target_size, time_limit):
        nonlocal start_time, deadline
        end_time = start_time + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        
        max_restarts = 100
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            # Create initial guess
            # Strategy: Start with prescribed, fill rest with randoms from a "spread" range
            # to minimize initial conflicts.
            current_set = list(prescribed)
            needed = target_size - len(current_set)
            
            if needed < 0:
                return None
                
            # Generate random candidates
            # To make the initial state better, we can try to pick numbers that are far apart
            # or just random. Random is fine for SA to fix.
            pool = [x for x in range(1, 285) if x not in fixed_elements]
            random.shuffle(pool)
            current_set.extend(pool[:needed])
            random.shuffle(current_set)
            
            # Identify fixed indices
            is_fixed = [x in fixed_elements for x in current_set]
            
            # Evaluate initial state
            is_valid, current_conflict = check_sidon(current_set)
            
            if is_valid:
                return current_set
            
            # Simulated Annealing with Swap/Replace moves
            # We will use a temperature schedule that allows escaping local minima.
            # Move: Pick a random variable index, pick a random new value.
            # To improve efficiency, we can also try "Swap": pick two variable indices, swap their values.
            # But Replace is usually sufficient if the pool is large.
            
            temp = 10.0
            min_temp = 0.01
            cooling_rate = 0.999
            max_steps = 50000  # Limit steps per restart to allow many restarts
            
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
                    
                # Check if new_val is already in the set (excluding idx)
                # If it is, conflict will spike, which is fine, SA handles it.
                
                current_set[idx] = new_val
                
                # Evaluate new state
                is_valid, new_conflict = check_sidon(current_set)
                
                delta = new_conflict - current_conflict
                
                if delta <= 0:
                    current_conflict = new_conflict
                    if is_valid:
                        return current_set
                else:
                    # Accept worse solution with probability
                    # P = exp(-delta / temp)
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

    # We have ~110s total.
    # Greedy took negligible time.
    # We have best_size (likely 15-18).
    # We want to maximize size.
    
    # Strategy:
    # Try to find size 20.
    # If fails, try 19.
    # If fails, try 18 (if best_size < 18).
    
    # Allocate time dynamically?
    # Let's give 60s to size 20.
    # 30s to size 19.
    # 15s to size 18.
    # 5s buffer.
    
    if best_size < 20:
        res = try_find_solution(20, 60)
        if res:
            return res

    if best_size < 19:
        res = try_find_solution(19, 30)
        if res:
            return res

    if best_size < 18:
        res = try_find_solution(18, 15)
        if res:
            return res

    # If nothing improved, return the best we have (the greedy one)
    return best_set

# EVOLVE-BLOCK-END
