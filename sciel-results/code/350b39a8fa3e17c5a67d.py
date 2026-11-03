"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import sys
import time

def construct(n_rows, k):
    """
    Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences a_j - a_i (i < j) are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows == 0 or k == 0:
        return []
        
    # Target: minimize max value subject to no duplicate differences.
    # We use Simulated Annealing to minimize conflicts and then max value.
    
    best_rows = None
    best_score = float('inf')
    best_max_val = float('inf')
    
    start_time = time.time()
    max_time = 110.0
    
    # Initial guess for the upper bound of values.
    # Counting bound is n_rows * k * (k+1) / 2.
    # For 4, 7: 4 * 28 = 112.
    # Let's start with a bound slightly higher to allow flexibility, e.g., 150.
    initial_bound = int(n_rows * k * (k + 1) / 2) + 50
    
    # Run multiple SA runs
    num_runs = 20
    time_per_run = max_time / num_runs
    
    for run in range(num_runs):
        if time.time() - start_time > max_time - 5:
            break
            
        # Initialize a random valid configuration (increasing rows)
        # We generate random increasing sequences within [0, initial_bound]
        rows = []
        for r in range(n_rows):
            # Generate k+1 distinct integers from 0..initial_bound
            # Must start with 0.
            # Pick k distinct integers from 1..initial_bound
            vals = [0] + sorted(random.sample(range(1, initial_bound + 1), k))
            rows.append(vals)
        
        current_max_val = max(max(row) for row in rows)
        
        # Run SA
        rows, final_max, final_conflicts = _simulated_annealing(rows, k, time_per_run, initial_bound)
        
        if final_conflicts == 0:
            if final_max < best_max_val:
                best_max_val = final_max
                best_rows = rows
                
    if best_rows is None:
        # Fallback: Greedy construction
        best_rows = _greedy_fallback(n_rows, k)
        
    return best_rows

def _calculate_conflicts(rows, k):
    """
    Returns (num_conflicts, max_val).
    num_conflicts is the number of differences that appear more than once.
    """
    diff_counts = {}
    max_val = 0
    for row in rows:
        if not row: continue
        max_val = max(max_val, max(row))
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diff_counts:
                    diff_counts[d] += 1
                else:
                    diff_counts[d] = 1
                    
    # Count how many differences have count > 1
    # Note: if a diff appears 3 times, it contributes 2 to the "excess" count?
    # Or just count the number of duplicates?
    # For SA, we want to minimize the number of "bad" pairs.
    # Let's define cost as sum(count(d) - 1 for d where count(d) > 1).
    # This is equivalent to (total_diffs - unique_diffs).
    total_diffs = 0
    unique_diffs = 0
    for c in diff_counts.values():
        total_diffs += c
        if c > 0:
            unique_diffs += 1
            
    conflicts = total_diffs - unique_diffs
    return conflicts, max_val

def _simulated_annealing(rows, k, time_limit, max_val_bound):
    """
    Optimizes rows to minimize conflicts, then max value.
    """
    current_rows = [list(r) for r in rows]
    current_conflicts, current_max = _calculate_conflicts(current_rows, k)
    
    best_rows = [list(r) for r in current_rows]
    best_conflicts, best_max = current_conflicts, current_max
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > T_min:
        # Pick a random row and a random index to modify
        # Index 0 is always 0, so we pick from 1..k
        r_idx = random.randint(0, len(current_rows) - 1)
        p_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][p_idx]
        
        # Propose a new value
        # We want to keep the row increasing.
        # Lower bound: current_rows[r_idx][p_idx-1] + 1
        # Upper bound: min(current_rows[r_idx][p_idx+1] - 1, max_val_bound) if p_idx < k else max_val_bound
        
        lb = current_rows[r_idx][p_idx - 1] + 1
        if p_idx < k:
            ub = min(current_rows[r_idx][p_idx + 1] - 1, max_val_bound)
        else:
            ub = max_val_bound
            
        if lb > ub:
            T *= cooling
            continue
            
        # Randomly pick a value in [lb, ub]
        # To allow larger jumps, sometimes pick from a wider range?
        # For now, local moves.
        new_val = random.randint(lb, ub)
        
        if new_val == old_val:
            continue
            
        # Apply move
        current_rows[r_idx][p_idx] = new_val
        
        new_conflicts, new_max = _calculate_conflicts(current_rows, k)
        
        # Cost function: Primary is conflicts, Secondary is max value
        # We want to minimize conflicts first.
        # If conflicts are equal, minimize max value.
        # If conflicts increase, we might still accept if T is high.
        
        # Let's define a combined cost.
        # Cost = conflicts * 1000 + max_val
        # This prioritizes eliminating conflicts.
        old_cost = current_conflicts * 1000 + current_max
        new_cost = new_conflicts * 1000 + new_max
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_conflicts = new_conflicts
            current_max = new_max
            
            # Update best
            if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_max < best_max):
                best_conflicts = new_conflicts
                best_max = new_max
                best_rows = [list(r) for r in current_rows]
        else:
            # Reject
            current_rows[r_idx][p_idx] = old_val
            
        T *= cooling
        
    return best_rows, best_max, best_conflicts

def _greedy_fallback(n_rows, k):
    """Simple greedy to ensure we return *something* valid."""
    rows = []
    used_diffs = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
