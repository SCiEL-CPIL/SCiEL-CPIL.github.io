import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    N = 1000
    MAX_SUM = 2 * N
    
    def run_greedy(noise_scale=0.0, seed=42):
        random.seed(seed)
        chosen = []
        # Using a list for O(1) access to sum counts
        sum_counts = [0] * (MAX_SUM + 1)
        chosen_set = set()
        
        while True:
            if time.time() > deadline:
                break
                
            best_x = None
            best_score = float('inf')
            best_new_sums = [] # Not needed if we update directly, but let's keep logic similar
            
            # Iterate through all candidates not in chosen
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                
                valid = True
                tight_count = 0
                
                # Check 2*x
                c_double = sum_counts[2 * x]
                if c_double >= 2:
                    valid = False
                elif c_double == 1:
                    tight_count += 1
                
                # Check x + y for y in chosen
                if valid:
                    for y in chosen:
                        c_sum = sum_counts[x + y]
                        if c_sum >= 2:
                            valid = False
                            break
                        elif c_sum == 1:
                            tight_count += 1
                
                if not valid:
                    continue
                
                # Score: prefer lower tight_count
                # Add noise
                score = tight_count + random.random() * noise_scale
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            # Add best_x
            chosen.append(best_x)
            chosen_set.add(best_x)
            
            # Update counts
            sum_counts[2 * best_x] += 1
            for y in chosen:
                if y != best_x:
                    sum_counts[best_x + y] += 1
        
        return sorted(chosen)

    def local_search(initial_set):
        """
        Try to improve the set by swapping elements.
        """
        chosen = list(initial_set)
        chosen_set = set(chosen)
        sum_counts = [0] * (MAX_SUM + 1)
        
        # Rebuild counts
        for x in chosen:
            sum_counts[2*x] += 1
            for y in chosen:
                if y > x:
                    sum_counts[x+y] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element
            added = False
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                valid = True
                c_double = sum_counts[2*x]
                if c_double >= 2:
                    valid = False
                else:
                    for y in chosen:
                        if sum_counts[x+y] >= 2:
                            valid = False
                            break
                if valid:
                    # Add it
                    chosen.append(x)
                    chosen_set.add(x)
                    sum_counts[2*x] += 1
                    for y in chosen:
                        if y != x:
                            sum_counts[x+y] += 1
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # If no addition, try swaps
            if not chosen:
                break
                
            # Pick a random element to remove
            rem_idx = random.randint(0, len(chosen) - 1)
            rem_val = chosen[rem_idx]
            
            # Temporarily remove
            chosen.pop(rem_idx)
            chosen_set.remove(rem_val)
            
            # Update counts (decrement)
            sum_counts[2*rem_val] -= 1
            for y in chosen:
                sum_counts[rem_val+y] -= 1
            
            # Find a valid addition
            found_add = False
            # Try all y, pick the first valid one (or best one?)
            # Let's try to find any valid addition
            for y in range(1, N + 1):
                if y in chosen_set:
                    continue
                valid = True
                c_double = sum_counts[2*y]
                if c_double >= 2:
                    valid = False
                else:
                    for z in chosen:
                        if sum_counts[y+z] >= 2:
                            valid = False
                            break
                
                if valid:
                    # Swap: add y
                    chosen.append(y)
                    chosen_set.add(y)
                    sum_counts[2*y] += 1
                    for z in chosen:
                        if z != y:
                            sum_counts[y+z] += 1
                    improved = True
                    found_add = True
                    break
                
            if not found_add:
                # Restore rem_val
                chosen.append(rem_val)
                chosen_set.add(rem_val)
                sum_counts[2*rem_val] += 1
                for y in chosen:
                    if y != rem_val:
                        sum_counts[rem_val+y] += 1

        return chosen

    best_set = []
    best_size = 0
    
    # Phase 1: Deterministic Greedy (Low Noise)
    # Try a few fixed seeds with low noise
    for seed in range(5):
        if time.time() > deadline:
            break
        s = run_greedy(noise_scale=0.0, seed=seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Phase 2: Randomized Greedy Restarts
    # Higher noise to explore different structures
    while time.time() < deadline:
        s = run_greedy(noise_scale=1.0, seed=random.randint(0, 10000))
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
        # Occasionally run local search on the best found so far
        if random.random() < 0.1 and best_set:
            improved_s = local_search(best_set)
            if len(improved_s) > best_size:
                best_size = len(improved_s)
                best_set = improved_s

    return best_set
