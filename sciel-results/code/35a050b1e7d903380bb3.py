import sys
import time
import random
import copy
import math

def construct(n_rows, k):
    # Parameters
    # n_rows = 4, k = 7 (from task description, though function takes args)
    # Each row has k+1 elements.
    
    start_time = time.time()
    timeout = 110.0
    
    best_max = float('inf')
    best_solution = None
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_max(rows):
        return max(max(r) for r in rows)

    def greedy_randomized(max_limit=200):
        # Generate a valid solution using greedy with some randomization
        rows = []
        used = set()
        for r_idx in range(n_rows):
            row = [0]
            last = 0
            for _ in range(k):
                found = False
                # Try values starting from last+1
                candidates = []
                for x in range(last + 1, max_limit):
                    # Check conflicts
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    if len(candidates) > 10:
                        break
                
                if not candidates:
                    # If no candidate found, extend range significantly
                    for x in range(last + 1, 500):
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                        if len(candidates) > 10:
                            break
                            
                if not candidates:
                    return None # Failed to construct
                
                # Pick a random candidate to ensure diversity in initial solutions
                # Usually picking the smallest is best for minimizing max, 
                # but for SA, diverse starts help.
                # Let's pick from the first 3 valid candidates.
                pick_idx = random.randint(0, min(2, len(candidates)-1))
                x = candidates[pick_idx]
                
                # Update used
                for a in row:
                    used.add(x - a)
                
                row.append(x)
                last = x
            rows.append(row)
        return rows

    def simulated_annealing(initial_rows, time_limit):
        current_rows = [r[:] for r in initial_rows]
        current_max = get_max(current_rows)
        
        # Initial temperature
        T = 50.0
        # Cooling rate
        alpha = 0.995
        min_T = 0.1
        
        # Precompute some bounds for efficiency? 
        # For now, simple checks.
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < time_limit and T > min_T:
            # Pick a random row and index
            r = random.randint(0, n_rows - 1)
            # Index 0 is fixed at 0. So 1 to k.
            i = random.randint(1, k)
            
            old_val = current_rows[r][i]
            
            # Determine valid range for new value
            min_val = current_rows[r][i-1] + 1
            max_val = current_rows[r][i+1] - 1 if i < k else current_max + 10 # Allow some growth
            
            if min_val >= max_val:
                T *= alpha
                continue
            
            # Generate new value
            # Bias towards smaller values to help minimize max
            # Random delta
            # If i == k, we really want to decrease it if possible.
            # Otherwise, random move.
            
            # Strategy: 
            # 50% chance: try to decrease (if i==k) or random
            # Let's just pick a random value in [min_val, max_val]
            # But biasing towards the lower end helps convergence to small max.
            # Use a distribution that favors smaller numbers.
            # E.g., new_val = min_val + int(random.expovariate(1.0 / 10.0))
            # But must be < max_val.
            
            # Simple approach: random int in range.
            # To help reduce max, if i == k, we can specifically try to decrease.
            if i == k:
                # Try to decrease
                # Pick a random value in [min_val, old_val - 1] if possible
                if old_val > min_val:
                    new_val = random.randint(min_val, old_val - 1)
                else:
                    # Can't decrease, try increase or keep
                    new_val = old_val
            else:
                # Internal element, random move
                # Range is [min_val, max_val]
                # Bias towards center or lower?
                # Just random for now.
                new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue
            
            # Check validity of the change
            # Only need to check differences involving the changed element in row r
            # and differences that were previously using the old value.
            
            # For simplicity and correctness first, check full validity.
            # Optimization: 
            # 1. Remove old differences from a set? No, we don't maintain a global set efficiently in SA.
            # 2. Just construct the candidate row and check full validity.
            
            candidate_rows = [row[:] for row in current_rows]
            candidate_rows[r][i] = new_val
            
            # Quick check: strict increase
            if not (candidate_rows[r][i-1] < new_val < (candidate_rows[r][i+1] if i < k else float('inf'))):
                T *= alpha
                continue
                
            # Full validity check
            if is_valid(candidate_rows):
                new_max = get_max(candidate_rows)
                delta_max = new_max - current_max
                
                # Accept if better
                if delta_max < 0:
                    current_rows = candidate_rows
                    current_max = new_max
                    no_improve_count = 0
                else:
                    # Accept worse with probability exp(-delta / T)
                    if delta_max == 0:
                        # Accept equal with some probability to allow wandering
                        if random.random() < 0.1:
                            current_rows = candidate_rows
                    elif random.random() < math.exp(-delta_max / T):
                        current_rows = candidate_rows
                        current_max = new_max
                    no_improve_count += 1
            else:
                # Invalid move, count as no improve
                no_improve_count += 1
            
            # Cool down
            T *= alpha
            
            # Restart if stuck too long
            if no_improve_count > max_no_improve:
                # We could restart SA here, but the main loop handles restarts.
                # Breaking early might be better to save time for a new seed.
                break

        return current_rows, current_max

    # Main loop
    while time.time() - start_time < timeout:
        # Generate a new random greedy solution
        sol = greedy_randomized()
        if sol is None:
            continue
        
        curr_max = get_max(sol)
        if curr_max < best_max:
            best_max = curr_max
            best_solution = [r[:] for r in sol]
        
        # Run simulated annealing to improve
        # Allocate remaining time / 10 to each SA attempt
        remaining = timeout - (time.time() - start_time)
        if remaining < 1:
            break
        
        # Give SA a chunk of time. 
        # If we have lots of time, give it more.
        # If little, give it less.
        # Let's aim for ~5-10 seconds per SA run if time permits, 
        # but at least 0.5s.
        sa_time = max(0.5, min(5.0, remaining * 0.2))
        
        improved_sol, improved_max = simulated_annealing(sol, start_time + sa_time)
        
        if improved_max < best_max:
            best_max = improved_max
            best_solution = [r[:] for r in improved_sol]

    return best_solution
