"""
Strategy: Local Search / Constraint Satisfaction
Instead of building rows sequentially (greedy), we generate many candidate rows
independently (or in small batches) and then use a local search to assign
distinct rows to the n_rows slots such that their difference sets are disjoint.
This decouples the row generation from the global packing constraint, allowing
for "dense" rows that might not fit in a greedy sequence but fit well together
in a different combination.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline: 140 seconds (10s safety margin from 150s limit)
    deadline = time.time() + 140.0
    
    # 1. Generate a baseline greedy solution to have a valid fallback
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_score = _max_entry(baseline)
    
    # If the problem is very small, return baseline immediately
    if n_rows <= 2 and k <= 3:
        return baseline

    # 2. Local Search Strategy
    # We maintain a pool of candidate rows.
    # Each candidate is a list of k+1 integers starting at 0.
    # We also pre-compute the set of differences for each candidate.
    
    # Phase A: Generate a diverse pool of candidates
    # We want rows that are "tight" (small max entry) but diverse in their difference patterns.
    # We'll generate candidates using a randomized greedy method with varying "greediness".
    
    pool = []
    pool_diffs = []
    
    # Add the greedy baseline rows to the pool first? 
    # Actually, let's generate a large pool of random valid rows.
    # A valid row is just a set of k+1 numbers. The "validity" against other rows is checked later.
    
    max_pool_size = 200
    target_max_entry = best_score + 10  # Allow slightly larger rows initially, search will shrink
    
    # Generate candidates
    for _ in range(max_pool_size):
        if time.time() > deadline - 10:
            break
            
        # Randomized greedy generation of a single row
        # We want to bias towards smaller numbers but allow some randomness.
        row = _generate_random_row(k, max_val=target_max_entry)
        if row is None:
            continue
            
        diffs = _get_diffs(row)
        # Optional: Filter out rows that are "too similar" to existing pool?
        # For simplicity, just add them.
        pool.append(row)
        pool_diffs.append(diffs)
        
        # If we hit the deadline, break
        if time.time() > deadline - 5:
            break

    # Ensure we have at least n_rows candidates
    if len(pool) < n_rows:
        # Fallback to greedy if pool is too small
        return best_rows

    # Phase B: Local Search to select n_rows disjoint rows
    # State: assignment of pool indices to row slots [0, n_rows)
    # Cost: Number of conflicts (overlapping differences) + Penalty for large max entry
    
    best_assignment = None
    best_conflicts = float('inf')
    best_max_entry = float('inf')
    
    # Initialize with a random assignment
    for restart in range(500):
        if time.time() > deadline - 2:
            break
            
        # Randomly select n_rows indices from the pool
        # To bias towards good solutions, we could sort pool by max entry and pick from top, 
        # but random selection explores more. Let's mix: 50% random, 50% top-ranked.
        
        indices = list(range(len(pool)))
        random.shuffle(indices)
        
        # If we want to bias, we could sort indices by max(pool[i]) and take first n_rows + some random
        # Let's just take a random subset for now to ensure diversity.
        current_indices = indices[:n_rows]
        
        # Check initial validity
        conflicts, max_entry = _check_conflicts(current_indices, pool_diffs, pool)
        
        if conflicts == 0:
            # Found a valid solution!
            candidate_rows = [pool[i] for i in current_indices]
            if max_entry < best_score:
                best_score = max_entry
                best_rows = candidate_rows
                if best_score <= 110: # Early exit if we beat the known best significantly
                    return best_rows
            # Even if not better, it's a valid solution. 
            # We can try to improve it via local search swaps.
            
        # Local Search: Hill Climbing on the assignment
        # Move: Swap one index in current_indices with one index NOT in current_indices
        # Or: Replace one index with a new random index from the pool
        
        current_best_conflicts = conflicts
        current_best_max = max_entry
        current_best_indices = current_indices.copy()
        
        no_improvement_count = 0
        max_iterations = 200
        
        for iter in range(max_iterations):
            if time.time() > deadline - 2:
                break
                
            if current_best_conflicts == 0:
                # We have a valid solution. Try to reduce max_entry.
                # Swap to find a valid solution with smaller max entry.
                found_improvement = False
                # Try swapping one row with another from the pool
                # We prefer swapping with rows that have smaller max entry
                
                # Sort pool indices by max entry to try "better" swaps first
                sorted_pool_indices = sorted(range(len(pool)), key=lambda i: max(pool[i]))
                
                for i_swap in range(n_rows):
                    if found_improvement:
                        break
                    for j_new in sorted_pool_indices:
                        if j_new in current_best_indices:
                            continue
                        # Try swapping i_swap with j_new
                        new_indices = current_best_indices.copy()
                        new_indices[i_swap] = j_new
                        new_conflicts, new_max = _check_conflicts(new_indices, pool_diffs, pool)
                        
                        if new_conflicts == 0 and new_max < current_best_max:
                            current_best_indices = new_indices
                            current_best_max = new_max
                            current_best_conflicts = 0
                            found_improvement = True
                            break
                        elif new_conflicts < current_best_conflicts:
                            # If we have conflicts, we first want to resolve them
                            pass
                            
                if not found_improvement:
                    # Try random swaps to escape local minima in max_entry (if conflicts=0)
                    # Or just break if we can't improve
                    if random.random() < 0.1:
                        # Random restart of the local search phase?
                        break
                    else:
                        break
                        
            else:
                # We have conflicts. Try to reduce them.
                found_improvement = False
                # Identify the "worst" row (most conflicts) and try to replace it
                # Or just try random swaps
                
                for i_swap in range(n_rows):
                    if found_improvement:
                        break
                    # Try swapping with a random index from the pool
                    for _ in range(10): # Try a few random swaps per row
                        j_new = random.choice(range(len(pool)))
                        if j_new in current_best_indices:
                            continue
                        new_indices = current_best_indices.copy()
                        new_indices[i_swap] = j_new
                        new_conflicts, new_max = _check_conflicts(new_indices, pool_diffs, pool)
                        
                        if new_conflicts < current_best_conflicts:
                            current_best_indices = new_indices
                            current_best_conflicts = new_conflicts
                            current_best_max = new_max
                            found_improvement = True
                            break
                            # Don't break outer loop immediately, try to improve more?
                            # Let's break to re-evaluate
                            
                if not found_improvement:
                    no_improvement_count += 1
                    if no_improvement_count > 20:
                        break
                else:
                    no_improvement_count = 0

        # End of local search for this restart
        
        # Update global best
        if current_best_conflicts == 0:
            candidate_rows = [pool[i] for i in current_best_indices]
            if current_best_max < best_score:
                best_score = current_best_max
                best_rows = candidate_rows
                if best_score <= 110:
                    return best_rows

    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _generate_random_row(k, max_val=200):
    """
    Generates a random valid row (strictly increasing, starts at 0) with max entry <= max_val.
    Uses a randomized greedy approach.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        # Pick a random next value > x
        # Bias towards smaller values? 
        # Let's pick uniformly from [x+1, min(x+10, max_val)]
        upper = min(x + 10, max_val)
        if upper <= x:
            return None
        next_x = random.randint(x + 1, upper)
        row.append(next_x)
        x = next_x
        
    if max(row) > max_val:
        return None
        
    return row

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _check_conflicts(indices, pool_diffs, pool):
    """
    Checks if the rows at the given indices have disjoint difference sets.
    Returns (number_of_conflicts, max_entry).
    """
    used = set()
    conflicts = 0
    max_entry = 0
    
    for idx in indices:
        row = pool[idx]
        max_entry = max(max_entry, max(row))
        diffs = pool_diffs[idx]
        
        # Count conflicts
        conflict_count = len(diffs & used)
        conflicts += conflict_count
        used |= diffs
        
    return conflicts, max_entry

def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)
# EVOLVE-BLOCK-END
