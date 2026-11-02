"""Improved program for difference triangle set (7,5) task."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all pairwise differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    """Return set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _compute_all_used(rows):
    """Compute union of all difference sets."""
    used = set()
    for r in rows:
        used |= _row_diffs_set(r)
    return used

def _global_max(rows):
    """Return maximum entry across all rows."""
    return max(max(r) for r in rows)

def _validate(rows, n_rows, k):
    """Check validity of solution."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            return False
        for d in diffs:
            if d in used:
                return False
            used.add(d)
    return True

def _greedy_init(n_rows, k, seed=42):
    """Build initial solution greedily with smallest valid values."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            found = False
            for x in range(row[-1] + 1, row[-1] + 500):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                # Check Sidon property (all diffs distinct)
                if len(new_diffs) != len([x - a for a in row]):
                    continue
                row.append(x)
                used |= new_diffs
                found = True
                break
            if not found:
                # Should not happen for reasonable parameters
                break
        rows.append(row)
    return rows

def _find_random_valid_value(row, forbidden, max_val, rng):
    """
    Find a random valid value to append to row.
    Returns the value or None if not found.
    """
    if not row:
        return 0
    
    start = row[-1] + 1
    if max_val is not None and max_val < start:
        return None
    
    # Collect all forbidden values
    # A value x is forbidden if {x - a for a in row} intersects forbidden
    # or if any two values in {x - a for a in row} are equal (impossible since a are distinct)
    
    # We need x - a not in forbidden for all a in row
    # So x not in {a + f for a in row, f in forbidden}
    
    # Build set of forbidden x values
    forbidden_x = set()
    for a in row:
        for f in forbidden:
            forbidden_x.add(a + f)
    
    # Also need x > row[-1] and x <= max_val
    # Find candidate values
    candidates = []
    for x in range(start, (max_val if max_val is not None else start + 200) + 1):
        if x in forbidden_x:
            continue
        # Check that x - a values are all distinct (they are, since a are distinct)
        # And not in forbidden (already checked via forbidden_x)
        candidates.append(x)
        if len(candidates) >= 20:
            break
    
    if not candidates:
        return None
    
    # Weight towards smaller values
    # Use exponential weighting: weight = exp(-x / scale)
    scale = 15
    weights = [math.exp(-x / scale) for x in candidates]
    total_w = sum(weights)
    r = rng.random() * total_w
    cum = 0
    for x, w in zip(candidates, weights):
        cum += w
        if r <= cum:
            return x
    return candidates[-1]

def _construct_random_row(forbidden, max_val, k, rng):
    """
    Construct a random valid row of length k+1 starting at 0.
    Returns the row or None.
    """
    row = [0]
    used_in_row = set()
    
    for _ in range(k):
        # Forbidden for this step: global forbidden + diffs already used in this row
        step_forbidden = forbidden | used_in_row
        
        x = _find_random_valid_value(row, step_forbidden, max_val, rng)
        if x is None:
            return None
        
        row.append(x)
        # Update used_in_row with new diffs
        for a in row[:-1]:
            used_in_row.add(x - a)
    
    return row

def _local_search_minimize(rows, n_rows, k, deadline, rng, time_limit_fraction=0.7):
    """
    Local search to minimize the global max.
    Uses row replacement with simulated annealing.
    """
    start_time = time.time()
    cur = [r[:] for r in rows]
    cur_val = _global_max(cur)
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    cur_used = _compute_all_used(cur)
    
    best = [r[:] for r in cur]
    best_val = cur_val
    best_used = _compute_all_used(best)
    
    T = 3.0
    T_min = 0.05
    cool = 0.997
    
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline * time_limit_fraction and no_improve < max_no_improve:
        ri = rng.randrange(n_rows)
        
        # Compute forbidden diffs for row ri
        forbidden = cur_used - cur_row_diffs[ri]
        
        # Try to find a new row
        # First try to improve: max < cur_val
        # Then try neutral: max == cur_val
        # Then allow worse: max < cur_val + 3
        
        new_row = None
        
        # Try improving
        new_row = _construct_random_row(forbidden, cur_val - 1, k, rng)
        if new_row is not None:
            new_max = max(max(cur[i]) for i in range(n_rows) if i != ri)
            new_max = max(new_max, max(new_row))
            delta = new_max - cur_val
            if delta < 0:
                # Accept
                old_diffs = cur_row_diffs[ri]
                cur_used = cur_used - old_diffs
                cur_row_diffs[ri] = _row_diffs_set(new_row)
                cur_used |= cur_row_diffs[ri]
                cur[ri] = new_row
                cur_val = new_max
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    best_used = _compute_all_used(best)
                    no_improve = 0
                else:
                    no_improve += 1
                T = max(T_min, T * cool)
                continue
        
        # Try neutral
        new_row = _construct_random_row(forbidden, cur_val, k, rng)
        if new_row is not None:
            new_max = max(max(cur[i]) for i in range(n_rows) if i != ri)
            new_max = max(new_max, max(new_row))
            delta = new_max - cur_val
            if delta <= 0:
                if delta == 0 and rng.random() < 0.3:
                    # Accept with some probability to allow exploration
                    old_diffs = cur_row_diffs[ri]
                    cur_used = cur_used - old_diffs
                    cur_row_diffs[ri] = _row_diffs_set(new_row)
                    cur_used |= cur_row_diffs[ri]
                    cur[ri] = new_row
                    cur_val = new_max
                    no_improve += 1
                T = max(T_min, T * cool)
                continue
        
        # Try slightly worse
        new_row = _construct_random_row(forbidden, cur_val + 3, k, rng)
        if new_row is not None:
            new_max = max(max(cur[i]) for i in range(n_rows) if i != ri)
            new_max = max(new_max, max(new_row))
            delta = new_max - cur_val
            if delta > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    old_diffs = cur_row_diffs[ri]
                    cur_used = cur_used - old_diffs
                    cur_row_diffs[ri] = _row_diffs_set(new_row)
                    cur_used |= cur_row_diffs[ri]
                    cur[ri] = new_row
                    cur_val = new_max
                    no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        # No valid row found
        no_improve += 1
        T = max(T_min, T * cool)
        
        # Restart from best if stuck
        if no_improve > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = _compute_all_used(best)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 3.0
            no_improve = 0
    
    return best, best_val

def _polish(rows, n_rows, k, deadline, rng):
    """
    Final polish: try to reduce each row's max individually.
    """
    best = [r[:] for r in rows]
    best_val = _global_max(best)
    best_used = _compute_all_used(best)
    
    # Multiple passes
    for _ in range(5):
        for ri in range(n_rows):
            forbidden = best_used - _row_diffs_set(best[ri])
            
            # Try to find a row with smaller max
            for target_max in range(best_val - 1, best_val - 5, -1):
                new_row = _construct_random_row(forbidden, target_max, k, rng)
                if new_row is not None:
                    # Verify validity
                    if _is_valid_row(new_row, forbidden, k):
                        other_max = max(max(best[i]) for i in range(n_rows) if i != ri)
                        new_max = max(other_max, max(new_row))
                        if new_max < best_val:
                            best[ri] = new_row
                            best_val = new_max
                            best_used = _compute_all_used(best)
                            break
    
    return best, best_val

def _is_valid_row(row, forbidden, k):
    """Check if row is valid given forbidden diffs."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    diffs = _row_diffs(row)
    if len(diffs) != len(set(diffs)):
        return False
    if set(diffs) & forbidden:
        return False
    return True

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    # Generate multiple initial solutions
    initial_solutions = []
    for seed in [42, 123, 456, 789, 1024, 2048]:
        rows = _greedy_init(n_rows, k, seed=seed)
        if _validate(rows, n_rows, k):
            initial_solutions.append(rows)
    
    # Pick the best initial solution
    if not initial_solutions:
        # Fallback
        rows = _greedy_init(n_rows, k, seed=42)
        return rows
    
    best_init = min(initial_solutions, key=lambda r: _global_max(r))
    best_val = _global_max(best_init)
    
    # Run local search from the best initial solution
    result, result_val = _local_search_minimize(
        best_init, n_rows, k, deadline, rng, time_limit_fraction=0.75
    )
    
    # Polish
    result, result_val = _polish(result, n_rows, k, deadline, rng)
    
    # Try a few more local searches from different starts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for sol in initial_solutions[1:]:
            if time.time() > deadline - 10:
                break
            temp_sol, temp_val = _local_search_minimize(
                sol, n_rows, k, deadline, rng, time_limit_fraction=0.3
            )
            temp_sol, temp_val = _polish(temp_sol, n_rows, k, deadline, rng)
            if temp_val < result_val:
                result = temp_sol
                result_val = temp_val
    
    # Final validation
    if not _validate(result, n_rows, k):
        result = _greedy_init(n_rows, k, seed=42)
    
    return result

# EVOLVE-BLOCK-END
