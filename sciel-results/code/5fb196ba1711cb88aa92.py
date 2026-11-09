"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct(n_rows, k):
    # Strategy: Simulated Annealing on the set of rows.
    # Objective: Minimize the maximum entry, subject to all differences being distinct.
    # We define an energy function that penalizes conflicts and the max value.
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Initialize with a valid solution using a greedy approach
    def _greedy_row(used, k):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row

    used = set()
    current_rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        current_rows.append(row)
    
    best_rows = [r[:] for r in current_rows]
    best_max = max(max(r) for r in best_rows)
    
    # Helper to calculate energy
    # Energy = Conflict_Penalty * (# conflicts) + Max_Penalty * (max_value)
    # We want 0 conflicts and low max.
    CONFLICT_PENALTY = 1000
    MAX_PENALTY = 1
    
    def get_energy(rows):
        all_diffs = []
        max_val = 0
        for r in rows:
            max_val = max(max_val, max(r))
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        
        # Count conflicts
        unique_diffs = set(all_diffs)
        n_conflicts = len(all_diffs) - len(unique_diffs)
        
        return n_conflicts * CONFLICT_PENALTY + max_val * MAX_PENALTY

    current_energy = get_energy(current_rows)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T = 100.0
    alpha = 0.999
    min_T = 0.01
    
    # Move: Pick a random row and a random index > 0, change value
    # To keep rows sorted and increasing, we must be careful.
    # A simpler move: Pick a row, pick an index i, change row[i] to a random value 
    # such that row[i-1] < new_val < row[i+1] (if exists).
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Generate a neighbor
        new_rows = [r[:] for r in current_rows]
        r_idx = random.randint(0, n_rows - 1)
        # Pick an index to change, not the first one (0)
        pos_idx = random.randint(1, k)
        
        # Determine valid range for the new value
        min_val = new_rows[r_idx][pos_idx - 1] + 1
        max_val = new_rows[r_idx][pos_idx + 1] - 1 if pos_idx < k else best_max + 10
        
        if min_val > max_val:
            T *= alpha
            continue
            
        # Randomly pick a new value in the range, biased towards current
        # Or just a small perturbation
        delta = random.randint(-5, 5)
        new_val = new_rows[r_idx][pos_idx] + delta
        
        if new_val < min_val:
            new_val = min_val
        if new_val > max_val:
            new_val = max_val
            
        if new_val == new_rows[r_idx][pos_idx]:
            T *= alpha
            continue
            
        new_rows[r_idx][pos_idx] = new_val
        
        # Check if the row is still strictly increasing
        # Since we picked from [prev+1, next-1], it is.
        
        new_energy = get_energy(new_rows)
        delta_E = new_energy - current_energy
        
        # Acceptance probability
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_rows = new_rows
            current_energy = new_energy
            
            # Check if we have a valid solution (0 conflicts)
            # We need to extract conflicts count separately or infer from energy
            # If energy < CONFLICT_PENALTY, conflicts are 0.
            if current_energy < CONFLICT_PENALTY:
                current_max = max(max(r) for r in current_rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
                    # Update time limit check or continue
                    # If we hit the lower bound (105), we can stop early
                    if best_max <= 105:
                        break
        
        T *= alpha

    return best_rows
# EVOLVE-BLOCK-END
