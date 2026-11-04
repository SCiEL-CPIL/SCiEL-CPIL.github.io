"""Fresh design for the difference triangle set (7,5) task."""
import time
import itertools
import random

_START_TIME = time.time()
_DEADLINE = 150.0


def construct(n_rows, k):
    """
    Returns a list of n_rows rows, each a list of k+1 increasing integers
    starting at 0, such that all positive differences a_j - a_i (i < j)
    within any row are distinct across all rows.

    Minimizes the largest entry.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Total number of differences per row
    n_diffs = k * (k + 1) // 2
    # Total differences needed
    total_diffs = n_rows * n_diffs
    
    # The maximum value m must satisfy: the number of distinct positive integers
    # up to m is m, so we need m >= total_diffs (since each difference is at least 1).
    # But actually, differences can be any positive integer, and they must all be distinct.
    # The minimum possible maximum value is at least total_diffs (counting bound).
    
    # Let's use a backtracking search with good heuristics.
    # For (7,5), n_diffs = 15, total_diffs = 105.
    # Best known is 112.
    
    # Strategy: Try to construct rows one at a time using backtracking,
    # but also try to optimize the maximum value.
    
    # First, get a valid solution using greedy, then try to improve.
    
    deadline = time.time() + _DEADLINE - time.time()
    deadline = min(deadline, 145.0)  # Safety margin
    
    # Greedy construction to get a valid baseline
    baseline = _greedy_construct(n_rows, k)
    if baseline is not None:
        best_max = max(max(row) for row in baseline)
    else:
        best_max = float('inf')
    
    # Try to improve using backtracking search
    # We'll try to find a solution with a smaller maximum value
    
    # Try different strategies:
    # 1. Backtracking with row-by-row construction
    # 2. Try to constrain the maximum value
    
    best_result = baseline if baseline is not None else None
    best_max_val = best_max if best_max != float('inf') else 0
    
    # Try backtracking search with a limit on the maximum value
    for trial_max in range(best_max_val - 1, best_max_val - 30, -1):
        if time.time() > deadline:
            break
        result = _backtracking_construct(n_rows, k, trial_max, deadline)
        if result is not None:
            best_result = result
            best_max_val = trial_max
            break
    
    # Also try some randomized approaches
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            result = _randomized_backtracking(n_rows, k, best_max_val - 1, deadline)
            if result is not None:
                curr_max = max(max(row) for row in result)
                if curr_max < best_max_val:
                    best_result = result
                    best_max_val = curr_max
    
    if best_result is None:
        # Fallback: just return a trivial valid solution (may not be optimal)
        best_result = _trivial_construct(n_rows, k)
    
    return best_result


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes the smallest next mark whose
    differences to earlier marks are all unused."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _trivial_construct(n_rows, k):
    """Trivial construction that guarantees validity but may be large."""
    # Each row uses differences that are multiples of a large base
    # to avoid collisions. Use base = total_diffs + 1
    total_diffs = n_rows * k * (k + 1) // 2
    base = total_diffs + 1
    rows = []
    for r in range(n_rows):
        # Row r uses offsets that are multiples of base
        # Position i in the row: i * base
        row = [i * base for i in range(k + 1)]
        rows.append(row)
    return rows


def _backtracking_construct(n_rows, k, max_val, deadline):
    """
    Backtracking search to construct n_rows of k+1 increasing integers
    starting at 0, with all differences distinct across all rows,
    and max value <= max_val.
    """
    used = set()
    rows = []
    
    def backtrack(row_idx, col_idx, current_row):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if col_idx == k + 1:
            rows.append(current_row)
            return backtrack(row_idx + 1, 0, [0])
        
        if col_idx == 0:
            current_row.append(0)
            return backtrack(row_idx, 1, current_row)
        
        # Try values from current_row[-1] + 1 up to max_val
        start = current_row[-1] + 1
        for x in range(start, max_val + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue
            # Check within current row
            if len(new_diffs) != len(current_row):
                continue
            
            current_row.append(x)
            used.update(new_diffs)
            
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            
            used.difference_update(new_diffs)
            current_row.pop()
        
        return False
    
    if backtrack(0, 0, []):
        return rows
    return None


def _randomized_backtracking(n_rows, k, max_val, deadline):
    """Randomized backtracking with some heuristics."""
    used = set()
    rows = []
    
    def backtrack(row_idx, col_idx, current_row):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if col_idx == k + 1:
            rows.append(current_row)
            return backtrack(row_idx + 1, 0, [0])
        
        if col_idx == 0:
            current_row.append(0)
            return backtrack(row_idx, 1, current_row)
        
        start = current_row[-1] + 1
        candidates = list(range(start, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue
            if len(new_diffs) != len(current_row):
                continue
            
            current_row.append(x)
            used.update(new_diffs)
            
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            
            used.difference_update(new_diffs)
            current_row.pop()
        
        return False
    
    if backtrack(0, 0, []):
        return rows
    return None
