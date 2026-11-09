"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _get_all_diffs(rows):
    """Get all positive differences from all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, rng):
    """Greedy construction to get a valid starting point."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, max_iter=50000, time_limit=10.0):
    """Simulated annealing to reduce the maximum entry."""
    start_time = time.time()
    current = [list(r) for r in rows]
    current_max = _max_entry(current)
    best = [list(r) for r in current]
    best_max = current_max
    
    # Temperature schedule
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random row and a random position to modify
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)  # Don't modify position 0 (always 0)
        
        # Try to change the value at this position
        old_val = current[row_idx][pos]
        # Propose a new value: either decrease or increase slightly
        if rng.random() < 0.7 and old_val > 0:
            # Try to decrease
            delta = rng.randint(1, min(5, old_val - (current[row_idx][pos-1] + 1)))
            if delta > 0:
                new_val = old_val - delta
            else:
                new_val = old_val + rng.randint(1, 3)
        else:
            new_val = old_val + rng.randint(1, 5)
        
        # Ensure increasing order
        if pos > 0 and new_val <= current[row_idx][pos-1]:
            continue
        if pos < k and new_val >= current[row_idx][pos+1]:
            continue
        
        # Apply change
        current[row_idx][pos] = new_val
        
        # Check validity and compute new max
        new_max = _max_entry(current)
        valid = _is_valid(current)
        
        if valid:
            if new_max < best_max:
                best = [list(r) for r in current]
                best_max = new_max
            # Accept if better or with probability
            if new_max <= current_max or rng.random() < np.exp(-(new_max - current_max) / T):
                current_max = new_max
            else:
                # Reject
                current[row_idx][pos] = old_val
        else:
            # Invalid: reject with some probability based on how bad it is
            # Count conflicts
            diffs = set()
            conflict_count = 0
            for row in current:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in diffs:
                            conflict_count += 1
                        diffs.add(d)
            # Reject invalid moves mostly
            if rng.random() > 0.05:
                current[row_idx][pos] = old_val
        
        T = max(T_min, T * cooling)
    
    return best, best_max

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all differences are distinct over all rows, minimizing
    the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = random.Random(42)
    
    # Get initial valid solution via greedy
    initial_rows = _greedy_construct(n_rows, k, rng)
    best_rows = [list(r) for r in initial_rows]
    best_max = _max_entry(best_rows)
    
    # Try multiple restarts with local search
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        
        # Create a perturbed version of the best solution
        perturbed = [list(r) for r in best_rows]
        
        # Randomly perturb some values
        for _ in range(rng.randint(1, 5)):
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)
            # Shift some values
            shift = rng.randint(-3, 3)
            if shift != 0:
                # Apply shift to this position and adjust to maintain order
                old_val = perturbed[row_idx][pos]
                new_val = old_val + shift
                if pos > 0 and new_val <= perturbed[row_idx][pos-1]:
                    new_val = perturbed[row_idx][pos-1] + 1
                if pos < k and new_val >= perturbed[row_idx][pos+1]:
                    new_val = perturbed[row_idx][pos+1] - 1
                if new_val > 0:
                    perturbed[row_idx][pos] = new_val
        
        # Check if perturbed is valid; if not, repair it
        if not _is_valid(perturbed):
            # Repair: try to fix conflicts by adjusting values
            for _ in range(100):
                if _is_valid(perturbed):
                    break
                # Find a conflict
                diffs = {}
                conflict = None
                for ri, row in enumerate(perturbed):
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            d = row[j] - row[i]
                            if d in diffs:
                                conflict = (ri, i, j, diffs[d])
                                break
                            diffs[d] = (ri, i, j)
                    if conflict:
                        break
                if conflict:
                    ri, i, j, prev = conflict
                    # Try to adjust one of the conflicting entries
                    pos_to_fix = rng.choice([i, j])
                    old_val = perturbed[ri][pos_to_fix]
                    # Try moving it
                    for delta in [1, -1, 2, -2, 3, -3]:
                        new_val = old_val + delta
                        if pos_to_fix > 0 and new_val <= perturbed[ri][pos_to_fix-1]:
                            continue
                        if pos_to_fix < k and new_val >= perturbed[ri][pos_to_fix+1]:
                            continue
                        if new_val < 0:
                            continue
                        perturbed[ri][pos_to_fix] = new_val
                        if _is_valid(perturbed):
                            break
                        perturbed[ri][pos_to_fix] = old_val
        
        if not _is_valid(perturbed):
            # Skip this restart if we can't make it valid
            continue
        
        perturbed_max = _max_entry(perturbed)
        
        # Run local search on this perturbed solution
        time_per_restart = min(10.0, (deadline - 5.0 - time.time()) / max(1, 3 - num_restarts))
        improved_rows, improved_max = _local_search(
            perturbed, n_rows, k, 
            max_iter=30000, 
            time_limit=time_per_restart
        )
        
        if improved_max < best_max:
            best_rows = improved_rows
            best_max = improved_max
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    return best_rows
# EVOLVE-BLOCK-END
