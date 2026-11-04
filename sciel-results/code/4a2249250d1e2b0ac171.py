"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs
    
    best_rows = None
    best_max = float('inf')
    
    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper to compute cost: 
    # We want to minimize max_entry, but we must ensure validity (distinct diffs).
    # Cost = max_entry + PENALTY * (number of duplicate differences)
    # If duplicates > 0, the solution is invalid. We want to drive duplicates to 0 
    # and then reduce max_entry.
    # A better cost: 
    # If valid: cost = max_entry
    # If invalid: cost = max_entry + LARGE_PENALTY * num_duplicates
    # But to allow SA to move between invalid states, we need a smoother landscape.
    # Let's use: cost = max_entry + CONFLICT_PENALTY * num_conflicts
    # where num_conflicts is the number of differences that appear more than once.
    
    def compute_cost(rows, max_limit):
        used_diffs = {}
        max_entry = 0
        for row in rows:
            if row:
                m = max(row)
                if m > max_entry:
                    max_entry = m
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        used_diffs[d] += 1
                    else:
                        used_diffs[d] = 1
        
        num_conflicts = 0
        for count in used_diffs.values():
            if count > 1:
                num_conflicts += count - 1 # Number of excess occurrences
        
        # Penalty should be large enough to prevent accepting invalid solutions 
        # unless it leads to a significant reduction in max_entry later, 
        # but small enough to allow exploration.
        # Actually, for minimization of max_entry subject to validity,
        # we can just maximize the penalty for conflicts.
        conflict_penalty = 10000 * num_conflicts
        
        # If max_entry exceeds max_limit, add penalty
        limit_penalty = 0
        if max_entry > max_limit:
            limit_penalty = 1000 * (max_entry - max_limit)
            
        return max_entry + conflict_penalty + limit_penalty

    def generate_random_rows(max_val):
        rows = []
        for _ in range(n_rows):
            # Generate k+1 increasing integers starting at 0
            # First is 0. Rest are chosen from 1..max_val
            # Must be strictly increasing.
            # Simple way: pick k distinct integers from 1..max_val, sort, prepend 0
            if max_val < k:
                # Fallback
                row = list(range(k+1))
            else:
                # Pick k distinct values from 1 to max_val
                vals = random.sample(range(1, max_val + 1), k)
                vals.sort()
                row = [0] + vals
            rows.append(row)
        return rows

    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Initial baseline
    baseline_rows = pure_greedy()
    best_rows = [row[:] for row in baseline_rows]
    best_max = get_max_entry(best_rows)
    
    # SA Parameters
    T_start = 100.0
    T_end = 1.0
    cooling_rate = 0.999
    
    # We will run SA multiple times with different initial bounds
    # Start with a bound slightly above the greedy solution
    current_bound = best_max
    
    # To improve, we try to find a solution with max < best_max
    # So we set max_limit in cost function to best_max - 1 to encourage smaller values
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize SA
        # Start with a random configuration within a range
        # If we have a good best_max, we can start near it
        init_max = best_max + 10
        current_rows = generate_random_rows(init_max)
        
        # If the initial random is already valid and better, update
        c = compute_cost(current_rows, best_max - 1)
        if c < best_max: # This implies valid (no penalty) and max < best_max
             # Check validity explicitly to be safe
             used = set()
             valid = True
             for row in current_rows:
                 for i in range(len(row)):
                     for j in range(i+1, len(row)):
                         d = row[j] - row[i]
                         if d in used:
                             valid = False
                             break
                         used.add(d)
                 if not valid: break
             if valid:
                 curr_max = get_max_entry(current_rows)
                 if curr_max < best_max:
                     best_max = curr_max
                     best_rows = [row[:] for row in current_rows]
        
        T = T_start
        current_cost = compute_cost(current_rows, best_max - 1)
        
        while T > T_end and time.time() < deadline:
            # Generate a neighbor
            # Pick a random row and a random position (1..k)
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            
            # Pick a new value for this position
            # It must be > previous element (if c_idx > 1) and < next element (if c_idx < k)
            # Also must be <= current_bound (which is best_max - 1 for improvement)
            # Actually, we can allow it to go up to best_max - 1 to keep it competitive
            # Or slightly above to allow exploration? 
            # Let's restrict to [1, best_max - 1] to focus on finding better solutions.
            
            min_val = current_rows[r_idx][c_idx - 1] + 1
            max_val = best_max - 1
            
            # Also must be < next element if exists
            if c_idx < k:
                max_val = min(max_val, current_rows[r_idx][c_idx + 1] - 1)
            
            if min_val > max_val:
                T *= cooling_rate
                continue
                
            new_val = random.randint(min_val, max_val)
            
            # If the value is the same, skip
            if new_val == current_rows[r_idx][c_idx]:
                T *= cooling_rate
                continue
                
            # Apply move
            old_val = current_rows[r_idx][c_idx]
            current_rows[r_idx][c_idx] = new_val
            
            new_cost = compute_cost(current_rows, best_max - 1)
            delta_cost = new_cost - current_cost
            
            # Acceptance probability
            if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
                current_cost = new_cost
                
                # Check if we found a valid solution better than best
                # We can check validity periodically or when cost is low
                # Since cost includes penalty, if cost < best_max, it must be valid and smaller
                if new_cost < best_max:
                    # Verify validity to be 100% sure before updating best
                    used = set()
                    valid = True
                    for row in current_rows:
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                d = row[j] - row[i]
                                if d in used:
                                    valid = False
                                    break
                                used.add(d)
                        if not valid: break
                    if valid:
                        curr_max = get_max_entry(current_rows)
                        if curr_max < best_max:
                            best_max = curr_max
                            best_rows = [row[:] for row in current_rows]
            else:
                # Reject
                current_rows[r_idx][c_idx] = old_val
                
            T *= cooling_rate
            
        # End of SA run
        
    return best_rows
# EVOLVE-BLOCK-END
