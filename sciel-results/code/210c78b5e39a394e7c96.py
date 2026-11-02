"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 115
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Greedy with different starting biases
    for seed in range(30):
        if time.time() >= deadline - 15:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    # Strategy 2: Simulated annealing from best
    if best_rows is not None:
        best_rows, best_scope = _simulated_annealing(best_rows, n_rows, k, deadline)
    
    # Strategy 3: More aggressive search with larger moves
    if time.time() < deadline - 25:
        for seed in range(10):
            if time.time() >= deadline - 15:
                break
            rows = _greedy_construct(n_rows, k, seed + 100)
            if rows is not None:
                rows, scope = _simulated_annealing(rows, n_rows, k, deadline)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
    
    # Strategy 4: Final local polish
    if best_rows is not None and time.time() < deadline - 10:
        best_rows, best_scope = _local_polish(best_rows, n_rows, k, deadline)
    
    return best_rows

def _greedy_construct(n_rows, k, seed=0):
    """Greedy construction trying to minimize scope."""
    rng = random.Random(seed)
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

def _scope(rows):
    return max(max(row) for row in rows)

def _compute_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs_union(rows):
    """Compute union of all differences across all rows."""
    all_d = set()
    for row in rows:
        all_d |= _compute_diffs(row)
    return all_d

def _simulated_annealing(rows, n_rows, k, deadline):
    """Simulated annealing to escape local minima."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    rng = random.Random(42)
    start_temp = 50.0
    end_temp = 0.1
    cooling_rate = 0.995
    
    # Precompute diffs for current
    row_diffs = [_compute_diffs(r) for r in current_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    temp = start_temp
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline - 10 and temp > end_temp and iterations < max_iterations:
        iterations += 1
        
        # Pick a random row and position to modify
        ri = rng.randrange(n_rows)
        row = current_rows[ri]
        if len(row) <= 1:
            continue
        
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        # Try a random delta
        delta = rng.randint(-15, 15)
        if delta == 0:
            continue
        new_val = old_val + delta
        
        # Check validity
        if new_val <= 0:
            continue
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue
        
        # Compute old diffs for this row
        old_diffs = _compute_diffs(row)
        
        # Apply change
        row[pos] = new_val
        new_diffs = _compute_diffs(row)
        
        # Check for internal duplicates
        expected_diff_count = len(row) * (len(row) - 1) // 2
        if len(new_diffs) != expected_diff_count:
            row[pos] = old_val
            continue
        
        # Check conflicts with other rows
        other_diffs = all_diffs - old_diffs
        if new_diffs & other_diffs:
            row[pos] = old_val
            continue
        
        # Valid move - compute new scope
        new_scope = _scope(current_rows)
        delta_scope = new_scope - current_scope
        
        # Accept if better, or with probability exp(-delta_scope/temp) if worse
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temp):
            # Accept the move
            all_diffs = (all_diffs - old_diffs) | new_diffs
            row_diffs[ri] = new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Reject the move
            row[pos] = old_val
        
        temp *= cooling_rate
    
    return best_rows, best_scope

def _local_polish(rows, n_rows, k, deadline):
    """Final local polish with small moves to find immediate improvements."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    rng = random.Random(123)
    
    row_diffs = [_compute_diffs(r) for r in best_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    # Try many small moves
    for _ in range(20000):
        if time.time() >= deadline - 5:
            break
        
        ri = rng.randrange(n_rows)
        row = best_rows[ri]
        if len(row) <= 1:
            continue
        
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        # Try small deltas
        for delta in range(-3, 4):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            old_diffs = _compute_diffs(row)
            row[pos] = new_val
            new_diffs = _compute_diffs(row)
            
            expected_diff_count = len(row) * (len(row) - 1) // 2
            if len(new_diffs) != expected_diff_count:
                row[pos] = old_val
                continue
            
            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                row[pos] = old_val
                continue
            
            new_scope = _scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diffs[ri] = new_diffs
                break
            else:
                row[pos] = old_val
    
    return best_rows, best_scope

# EVOLVE-BLOCK-END
