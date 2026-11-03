"""Improved difference triangle set constructor with local search and backtracking."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    
    All positive differences within rows must be distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # First, get a valid solution using greedy
    rows = _greedy_construct(n_rows, k)
    best_scope = _get_scope(rows)
    best_rows = [row[:] for row in rows]
    
    if time.time() > deadline:
        return best_rows
    
    # Try to improve via local search
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random restart with perturbation
        perturbed = _perturb(best_rows, k, n_rows)
        if perturbed is None:
            continue
        
        improved = _local_search(perturbed, k, n_rows, deadline)
        if improved is not None:
            scope = _get_scope(improved)
            if scope < best_scope:
                best_scope = scope
                best_rows = [row[:] for row in improved]
        
        if time.time() > deadline:
            break
    
    # Final attempt: backtracking from best solution
    if time.time() < deadline - 20:
        bt_result = _backtrack_improve(best_rows, k, n_rows, deadline)
        if bt_result is not None:
            scope = _get_scope(bt_result)
            if scope < best_scope:
                best_scope = scope
                best_rows = [row[:] for row in bt_result]
    
    return best_rows


def _get_scope(rows):
    """Get the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes smallest next mark with unused differences."""
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


def _get_row_diffs(row):
    """Get all positive differences within a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_valid(rows):
    """Check if all differences across rows are distinct."""
    seen = set()
    for row in rows:
        for d in _get_row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _perturb(rows, k, n_rows):
    """Create a perturbed copy of rows by slightly modifying some entries."""
    perturbed = [row[:] for row in rows]
    # Randomly adjust some non-zero, non-max entries
    for _ in range(random.randint(1, min(3, n_rows))):
        ri = random.randint(0, n_rows - 1)
        row = perturbed[ri]
        # Pick a position to modify (not 0 or last)
        if len(row) > 2:
            pos = random.randint(1, len(row) - 2)
            delta = random.choice([-2, -1, 1, 2, 3])
            new_val = row[pos] + delta
            if new_val > row[pos - 1] and new_val < row[pos + 1]:
                row[pos] = new_val
    
    # Verify validity
    if not _all_diffs_valid(perturbed):
        return None
    return perturbed


def _local_search(rows, k, n_rows, deadline):
    """Local search: try to reduce scope by adjusting individual elements."""
    current = [row[:] for row in rows]
    current_scope = _get_scope(current)
    
    for iteration in range(200):
        if time.time() > deadline:
            return None
        
        improved = False
        # Try each row
        for ri in range(n_rows):
            if time.time() > deadline:
                break
            
            row = current[ri]
            row_max = max(row)
            
            # Try to reduce the maximum element
            if row[-1] > current_scope - 1:
                # Try moving last element down
                for new_val in range(row[-2] + 1, row[-1]):
                    test_row = row[:-1] + [new_val]
                    if _row_diffs_unique_with_others(test_row, current, ri):
                        current[ri] = test_row
                        new_scope = _get_scope(current)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                        break
            
            # Try moving other elements to create room
            for pos in range(1, len(row) - 1):
                for delta in [-1, 1]:
                    new_val = row[pos] + delta
                    if new_val <= row[pos - 1] or new_val >= row[pos + 1]:
                        continue
                    test_row = row[:]
                    test_row[pos] = new_val
                    if _row_diffs_unique_with_others(test_row, current, ri):
                        current[ri] = test_row
                        new_scope = _get_scope(current)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                        break
        
        if not improved:
            break
    
    return current


def _row_diffs_unique_with_others(row, all_rows, exclude_idx):
    """Check if row's diffs don't conflict with other rows' diffs."""
    row_diffs = _get_row_diffs(row)
    for i, other_row in enumerate(all_rows):
        if i == exclude_idx:
            continue
        other_diffs = _get_row_diffs(other_row)
        if row_diffs & other_diffs:
            return False
    return True


def _backtrack_improve(rows, k, n_rows, deadline):
    """Try backtracking to find a tighter solution."""
    best = [row[:] for row in rows]
    best_scope = _get_scope(best)
    
    # Try to rebuild each row with a tighter constraint
    for ri in range(n_rows):
        if time.time() > deadline:
            break
        
        # Get all diffs used by other rows
        other_diffs = set()
        for i, row in enumerate(best):
            if i != ri:
                other_diffs |= _get_row_diffs(row)
        
        # Try to find a better row for position ri
        new_row = _find_tighter_row(k, other_diffs, best[ri][-1] - 1, deadline)
        if new_row is not None and max(new_row) < best[ri][-1]:
            test_rows = [r[:] for r in best]
            test_rows[ri] = new_row
            if _all_diffs_valid(test_rows):
                best = test_rows
                best_scope = _get_scope(best)
    
    return best


def _find_tighter_row(k, forbidden_diffs, max_allowed, deadline):
    """Find a row of k+1 elements starting at 0 with max <= max_allowed,
    avoiding forbidden_diffs."""
    row = [0]
    x = 0
    
    while len(row) < k + 1:
        if time.time() > deadline:
            return None
        x += 1
        if x > max_allowed:
            return None
        new_diffs = {x - a for a in row}
        if not new_diffs & forbidden_diffs:
            row.append(x)
    
    return row
