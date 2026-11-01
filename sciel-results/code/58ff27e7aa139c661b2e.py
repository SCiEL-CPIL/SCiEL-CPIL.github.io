"""Difference triangle set (4,7): backtracking with heuristic + local search."""
import time
import random
import sys

sys.setrecursionlimit(10000)

_DEADLINE = 140.0  # safety margin before 150s kill
_start_time = None

def _time_left():
    if _start_time is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _start_time)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _validate(rows, k):
    """Check validity: each row has k+1 elements, starts at 0, strictly increasing,
    all positive differences distinct across all rows."""
    if len(rows) == 0:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    """Largest entry across all rows."""
    return max(max(row) for row in rows) if rows else 0

def _backtrack_row(row, k, used, max_val, deadline_fn):
    """Try to complete a row starting with current 'row' (list starting at 0),
    choosing next elements greedily with backtracking. Returns completed row or None."""
    if len(row) == k + 1:
        return row[:]
    
    last = row[-1]
    # Try candidates from last+1 upward
    for x in range(last + 1, max_val + 1):
        if deadline_fn() < 0.5:
            return None
        new_diffs = set()
        conflict = False
        for a in row:
            d = x - a
            if d in used or d in new_diffs:
                conflict = True
                break
            new_diffs.add(d)
        if not conflict:
            row.append(x)
            used_update = new_diffs
            result = _backtrack_row(row, k, used | used_update, max_val, deadline_fn)
            if result is not None:
                return result
            row.pop()
    return None

def _build_greedy(n_rows, k, max_val, seed=0):
    """Greedy construction with a given max_val. Returns rows or None if impossible."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            for x in range(x + 1, max_val + 1):
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _local_search(rows, k, max_val, time_budget, seed=0):
    """Try to reduce scope via local search: pick a random row, try to rebuild it
    with smaller max entry."""
    rng = random.Random(seed)
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    deadline = time.time() + time_budget
    
    while time.time() < deadline:
        # Pick a random row to try to improve
        idx = rng.randint(0, len(best_rows) - 1)
        current_row = best_rows[idx]
        current_max = current_row[-1]
        
        # Try to rebuild this row with a smaller max
        # Remove this row's differences from used set
        other_rows = [r for i, r in enumerate(best_rows) if i != idx]
        other_used = set()
        for r in other_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    other_used.add(r[j] - r[i])
        
        # Try to build a new row with max < current_max
        for target_max in range(current_max - 1, 0, -1):
            if time.time() > deadline:
                break
            new_row = _backtrack_row([0], k, other_used, target_max, 
                                     lambda: deadline - time.time())
            if new_row is not None:
                best_rows[idx] = new_row
                best_scope = _scope(best_rows)
                break
    
    return best_rows, best_scope

def _backtracking_search(n_rows, k, max_val, time_budget, seed=0):
    """Full backtracking search to find a valid configuration with scope <= max_val."""
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    
    # Build rows one at a time with backtracking
    used = set()
    rows = []
    
    def build_row(row_idx, current_row, remaining_max):
        if time.time() > deadline:
            return None
        if len(current_row) == k + 1:
            return current_row[:]
        last = current_row[-1]
        # Heuristic: try smaller gaps first, but also try some larger ones
        candidates = list(range(last + 1, remaining_max + 1))
        # Sort by how "tight" the differences are (prefer smaller new diffs)
        candidates.sort(key=lambda x: max(x - a for a in current_row))
        
        for x in candidates:
            if time.time() > deadline:
                return None
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue
            current_row.append(x)
            result = build_row(row_idx, current_row, remaining_max)
            if result is not None:
                return result
            current_row.pop()
        return None
    
    # Build each row
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = build_row(r, [0], max_val)
        if row is None:
            return None
        rows.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    return rows

def construct(n_rows, k):
    global _start_time
    _start_time = time.time()
    
    # First, get a baseline greedy solution
    # Estimate a reasonable max_val: greedy tends to give around k*(k+1)/2 * n_rows / something
    # For (4,7): k=7, n_rows=4, greedy gives around 150-200
    # Let's start with a generous max and try to reduce
    
    # Initial greedy with large max
    initial_max = 300
    best_rows = _build_greedy(n_rows, k, initial_max, seed=42)
    if best_rows is None:
        # Fallback: even more generous
        best_rows = _build_greedy(n_rows, k, 500, seed=42)
    
    best_scope = _scope(best_rows)
    
    # Try to improve via backtracking with tighter bounds
    # Binary search on max_val
    low = 100  # counting bound is 112 for (4,7)
    high = best_scope
    
    # First, try to find any solution with scope <= high via backtracking
    # Then try to reduce
    
    # Try backtracking search with progressively tighter bounds
    for attempt_max in range(high, low - 1, -1):
        if _time_left() < 5.0:
            break
        time_budget = min(_time_left() - 2.0, 10.0)
        if time_budget < 1.0:
            break
        result = _backtracking_search(n_rows, k, attempt_max, time_budget, seed=attempt_max)
        if result is not None:
            if _scope(result) < best_scope:
                best_rows = result
                best_scope = _scope(result)
                high = best_scope - 1
                # Try to improve further with local search
                if _time_left() > 10.0:
                    improved, imp_scope = _local_search(best_rows, k, best_scope, 
                                                         min(_time_left() - 5.0, 20.0), seed=123)
                    if imp_scope < best_scope:
                        best_rows = improved
                        best_scope = imp_scope
                        high = best_scope - 1
            else:
                # Same scope, but maybe we can try local search
                if _time_left() > 15.0:
                    improved, imp_scope = _local_search(best_rows, k, best_scope,
                                                         min(_time_left() - 5.0, 25.0), seed=456)
                    if imp_scope < best_scope:
                        best_rows = improved
                        best_scope = imp_scope
                        high = best_scope - 1
            break  # Found a solution at this max, stop binary search
    
    # If we still have time, try local search on the best solution
    if _time_left() > 8.0:
        improved, imp_scope = _local_search(best_rows, k, best_scope,
                                             min(_time_left() - 3.0, 30.0), seed=789)
        if imp_scope < best_scope:
            best_rows = improved
            best_scope = imp_scope
    
    # Validate final result
    if not _validate(best_rows, k):
        # Fallback to greedy
        best_rows = _build_greedy(n_rows, k, 500, seed=0)
    
    return best_rows
