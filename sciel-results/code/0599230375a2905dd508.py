"""Difference triangle set (7,5) - structured backtracking with incremental diff tracking."""
import time
import random
import sys

_DEADLINE = 148.0  # safety margin before 150s kill

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _valid(rows, k):
    """Check that all positive differences are distinct across all rows."""
    used = set()
    for row in rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used or d in diffs:
                    return False
                diffs.add(d)
        used |= diffs
    return True

def _fast_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with unused diffs."""
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

def _build_diff_counts(rows):
    """Build a dict mapping each difference to the row index that uses it."""
    diff_to_row = {}
    for idx, row in enumerate(rows):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_to_row[d] = idx
    return diff_to_row

def _remove_row_diffs(diff_to_row, row):
    """Remove all differences of a row from diff_to_row."""
    to_remove = []
    for d, idx in diff_to_row.items():
        # We don't track which row, just check if d is in this row's diffs
        pass
    # Better: rebuild by removing this row's diffs
    row_diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            row_diffs.add(row[j] - row[i])
    for d in row_diffs:
        if d in diff_to_row:
            del diff_to_row[d]

def _add_row_diffs(diff_to_row, row, row_idx):
    """Add all differences of a row to diff_to_row."""
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_to_row[d] = row_idx

def _row_rebuild(row, k, diff_to_row, exclude_row_idx, time_limit, rng, max_val):
    """
    Try to rebuild a single row given the fixed differences used by other rows.
    Returns a new valid row or None.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Differences that are already used by other rows
    forbidden = set(diff_to_row.keys())
    
    row = [0]
    used_in_row = set()
    
    def _place_element(row, used_in_row, pos):
        if pos == k + 1:
            return row[:]
        
        if time.time() > deadline:
            return None
        
        start_val = row[-1] + 1
        end_val = min(max_val, start_val + 100)
        
        candidates = list(range(start_val, end_val + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            new_diffs = set()
            valid = True
            for prev in row:
                d = val - prev
                if d in forbidden or d in used_in_row:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            row.append(val)
            used_in_row.update(new_diffs)
            
            result = _place_element(row, used_in_row, pos + 1)
            row.pop()
            
            if result is not None:
                return result
            
            used_in_row.difference_update(new_diffs)
        
        return None
    
    return _place_element(row, used_in_row, 1)

def _sa_optimize(rows, n_rows, k, time_limit, rng):
    """
    Simulated annealing: iteratively rebuild random rows to reduce scope.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_rows = rows[:]
    best_scope = _scope(rows)
    current_rows = rows[:]
    current_scope = best_scope
    
    temperature = 10.0
    cooling = 0.999
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        
        # Pick a random row to rebuild
        row_idx = rng.randint(0, n_rows - 1)
        
        # Build forbidden set from all other rows
        forbidden = set()
        for i, row in enumerate(current_rows):
            if i == row_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    forbidden.add(row[b] - row[a])
        
        # Try to rebuild the row with a max_val constraint
        # Try several max_val values
        found = None
        for max_val in range(current_scope, max(current_scope - 5, 100) - 1, -1):
            new_row = _row_rebuild(current_rows[row_idx], k, forbidden, row_idx, 
                                   min(0.5, (deadline - time.time()) * 0.1), rng, max_val)
            if new_row is not None:
                found = new_row
                break
        
        if found is None:
            # Try with a slightly larger max_val
            max_val = min(current_scope + 5, 150)
            found = _row_rebuild(current_rows[row_idx], k, forbidden, row_idx,
                                 min(0.5, (deadline - time.time()) * 0.1), rng, max_val)
        
        if found is None:
            continue
        
        # Compute new scope
        new_rows = current_rows[:]
        new_rows[row_idx] = found
        new_scope = _scope(new_rows)
        
        # Accept if better, or with probability exp(-(new_scope - current_scope)/temperature)
        if new_scope < current_scope:
            current_rows = new_rows
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = new_rows[:]
        elif new_scope == current_scope:
            # Accept equal with some probability for exploration
            if rng.random() < 0.3:
                current_rows = new_rows
        else:
            # Worse: accept with SA probability
            delta = new_scope - current_scope
            if rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
                current_rows = new_rows
                current_scope = new_scope
        
        temperature *= cooling
        temperature = max(temperature, 0.01)
        
        # If we hit the lower bound, stop
        if best_scope <= 105:
            break
    
    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all differences distinct across all rows.
    """
    start_time = time.time()
    
    # Quick greedy baseline
    greedy_rows = _fast_greedy(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    
    if not _valid(greedy_rows, k):
        # Should not happen, but fallback
        return greedy_rows
    
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Time budget for SA optimization
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed
    if remaining < 5:
        return best_rows
    
    # Run SA optimization with multiple restarts
    sa_time = min(remaining - 3, 100)
    
    # Try a few different random seeds for diversity
    for seed in [42, 123, 7, 999, 2025, 555, 777, 1234]:
        if time.time() > _DEADLINE - 3:
            break
        
        rng = random.Random(seed)
        # Start from greedy or from best found
        start_rows = best_rows[:] if seed != 42 else greedy_rows[:]
        
        # Give each SA run a portion of the time
        time_per_run = sa_time / 8.0
        result = _sa_optimize(start_rows, n_rows, k, time_per_run, rng)
        
        if result is not None and _valid(result, k):
            scope = _scope(result)
            if scope < best_scope:
                best_scope = scope
                best_rows = result[:]
        
        # If we hit the theoretical lower bound, stop
        if best_scope <= 105:
            break
    
    return best_rows
