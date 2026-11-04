"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return set of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used

def _greedy_row(used, k):
    """
    Greedily construct a row of k+1 integers starting at 0.
    Each new element is the smallest integer > last that doesn't create
    a difference already in 'used'.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs

def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= new_diffs
    return rows

def _randomized_greedy_row(used, k, rng, pool_size=5):
    """
    Randomized greedy: at each step, pick from the top 'pool_size' smallest
    valid candidates with some randomness.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.6:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, new_diffs

def _randomized_greedy(n_rows, k, rng, pool_size=5, deadline=None):
    """Randomized greedy construction with multiple restarts."""
    best_rows = None
    best_scope = float('inf')
    
    max_restarts = 100
    if deadline is None:
        deadline = time.time() + 100
        
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row, new_diffs = _randomized_greedy_row(used, k, rng, pool_size)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= new_diffs
        if success:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)

def _is_valid(rows):
    """Check if all positive differences across all rows are distinct."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _try_shrink_element(rows, used, row_idx, elem_idx, max_shrink=50):
    """
    Try to shrink rows[row_idx][elem_idx] by up to max_shrink.
    Returns True if a valid shrink was found and applied.
    """
    row = rows[row_idx]
    old_val = row[elem_idx]
    
    # Determine bounds
    if elem_idx > 0:
        lower_bound = row[elem_idx - 1] + 1
    else:
        lower_bound = 0  # First element is always 0, can't shrink
    
    if elem_idx < len(row) - 1:
        upper_bound = row[elem_idx + 1] - 1
    else:
        upper_bound = old_val
    
    # Only try to shrink if elem_idx > 0 (first element is fixed at 0)
    if elem_idx == 0:
        return False
    
    # Search for smaller valid values
    for new_val in range(old_val - 1, max(lower_bound, old_val - max_shrink) - 1, -1):
        # Calculate old and new diffs
        old_diffs = set()
        new_diffs = set()
        for i in range(len(row)):
            if i == elem_idx:
                continue
            if i < elem_idx:
                old_diffs.add(old_val - row[i])
                new_diffs.add(new_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
                new_diffs.add(row[i] - new_val)
        
        # Check conflict: new_diffs must not intersect with (used - old_diffs)
        conflict = False
        for d in new_diffs:
            if d in used and d not in old_diffs:
                conflict = True
                break
        
        if not conflict:
            # Valid move
            row[elem_idx] = new_val
            used = (used - old_diffs) | new_diffs
            return True
    
    return False

def _local_search_shrink(rows, k, rng, deadline):
    """
    Improved local search: tries to shrink multiple elements across all rows.
    Uses backtracking to escape local optima.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    stagnation_count = 0
    max_stagnation = 2000
    
    while time.time() < deadline and stagnation_count < max_stagnation:
        improved = False
        
        # Strategy 1: Try to shrink the largest element across all rows
        # Sort rows by their last element (descending)
        row_order = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)
        
        for r_idx in row_order:
            row = rows[r_idx]
            # Try to shrink from the last element backwards
            for e_idx in range(len(row) - 1, 0, -1):
                if _try_shrink_element(rows, used, r_idx, e_idx, max_shrink=30):
                    new_scope = _total_scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Strategy 2: Try random element shrinks
            for _ in range(20):
                if time.time() >= deadline:
                    break
                r_idx = rng.randint(0, len(rows) - 1)
                e_idx = rng.randint(1, len(rows[r_idx]) - 1)
                if _try_shrink_element(rows, used, r_idx, e_idx, max_shrink=15):
                    new_scope = _total_scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                    improved = True
                    break
        
        if not improved:
            stagnation_count += 1
            # Strategy 3: Perturbation to escape local optimum
            if stagnation_count % 50 == 0:
                r_idx = rng.randint(0, len(rows) - 1)
                e_idx = rng.randint(1, len(rows[r_idx]) - 1)
                old_val = rows[r_idx][e_idx]
                
                # Try small increases to change the difference landscape
                for delta in [1, 2, 3, 5, 8]:
                    new_val = old_val + delta
                    if new_val <= rows[r_idx][e_idx - 1]:
                        continue
                    if e_idx < len(rows[r_idx]) - 1 and new_val >= rows[r_idx][e_idx + 1]:
                        continue
                    
                    old_diffs = set()
                    new_diffs = set()
                    for i in range(len(rows[r_idx])):
                        if i == e_idx:
                            continue
                        if i < e_idx:
                            old_diffs.add(old_val - rows[r_idx][i])
                            new_diffs.add(new_val - rows[r_idx][i])
                        else:
                            old_diffs.add(rows[r_idx][i] - old_val)
                            new_diffs.add(rows[r_idx][i] - new_val)
                    
                    conflict = False
                    for d in new_diffs:
                        if d in used and d not in old_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        rows[r_idx][e_idx] = new_val
                        used = (used - old_diffs) | new_diffs
                        stagnation_count = 0
                        improved = True
                        break

    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        # Fallback: very simple construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = rows
    best_scope = _total_scope(rows)
    
    # Deadline management
    # We have 120s total. We want to leave 10s for safety.
    main_deadline = time.time() + 110
    
    # Phase 1: Randomized Greedy Restarts (15% of time)
    rg_deadline = time.time() + (110 * 0.15)
    
    rand_rows = _randomized_greedy(n_rows, k, rng, pool_size=5, deadline=rg_deadline)
    if rand_rows is not None:
        scope = _total_scope(rand_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rand_rows]
    
    # Phase 2: Local Search (85% of time)
    ls_deadline = min(time.time() + (110 * 0.85), main_deadline)
    
    if time.time() < ls_deadline:
        ls_rows = _local_search_shrink(best_rows, k, rng, ls_deadline)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [r[:] for r in ls_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
