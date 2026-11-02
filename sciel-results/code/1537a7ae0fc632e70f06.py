"""Model-and-optimize approach for difference triangle set (7,5).

Uses bitmask-based backtracking for efficient constraint checking.
For n_rows=7, k=5, the best known is 112.
"""

import time
import sys
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Special handling for the known instance (7,5)
    if n_rows == 7 and k == 5:
        result = _solve_7_5(start_time, deadline)
        if result is not None:
            return result
    
    # General fallback: greedy
    result = _greedy_construct(n_rows, k)
    return result


def _get_differences(row):
    """Get all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row by picking the smallest next mark
    that doesn't conflict with any previously used difference."""
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


def _verify_solution(rows, n_rows, k):
    """Verify that the solution is valid."""
    if len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        diffs = _get_differences(row)
        if diffs & all_diffs:
            return False
        all_diffs |= diffs
    return True


def _solve_7_5(start_time, deadline):
    """Solve (7,5) using bitmask-accelerated backtracking."""
    n_rows = 7
    k = 5
    target_size = k + 1  # 6 marks per row
    max_diff = 120  # differences up to 120 fit in a 128-bit mask
    
    # Use an integer bitmask for used differences (diffs 1..max_diff)
    # bit d set means difference d is used
    
    def _check_and_add_diffs(row, used_mask, max_mark):
        """Check if row's diffs conflict with used_mask; return new mask or -1."""
        # Compute all pairwise diffs in row
        row_diffs_mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d > max_diff:
                    return -1
                bit = 1 << d
                if row_diffs_mask & bit:
                    return -1  # internal conflict
                row_diffs_mask |= bit
        if row_diffs_mask & used_mask:
            return -1
        return used_mask | row_diffs_mask
    
    def _backtrack(row_idx, used_mask, rows, max_mark):
        """Build row row_idx, then recurse to next row."""
        if time.time() > deadline - 3:
            return None
        if row_idx == n_rows:
            return rows[:]
        
        # Build current row: start with [0], pick 5 more marks
        row = [0]
        return _build_row(row, row_idx, used_mask, rows, max_mark)
    
    def _build_row(row, row_idx, used_mask, rows, max_mark):
        """Recursively extend row until it has target_size elements."""
        if time.time() > deadline - 3:
            return None
        
        if len(row) == target_size:
            # Row complete - verify no conflicts
            new_mask = _check_and_add_diffs(row, used_mask, max_diff)
            if new_mask == -1:
                return None
            
            rows.append(row[:])
            if row_idx + 1 == n_rows:
                return rows[:]
            
            result = _backtrack(row_idx + 1, new_mask, rows, max_mark)
            if result is not None:
                return result
            
            rows.pop()
            return None
        
        last = row[-1]
        remaining = target_size - len(row)
        
        # Pruning: need at least (remaining-1) more marks after this one
        # Each must be at least 1 apart, so max_mark - last >= remaining
        if max_mark - last < remaining:
            return None
        
        # Determine candidate range
        start = last + 1
        # We need (remaining - 1) more marks after x, each >= 1 apart
        # So x <= max_mark - (remaining - 1)
        end = max_mark - (remaining - 1)
        
        if start > end:
            return None
        
        # Try candidates; use ordering that helps find solutions faster
        # For early rows, try smaller values first; for later, try larger first
        # to leave room for earlier rows' constraints
        if row_idx < 3:
            candidates = range(start, end + 1)
        else:
            candidates = range(end, start - 1, -1)
        
        for x in candidates:
            if time.time() > deadline - 3:
                return None
            
            # Quick check: new diffs x-a for a in row
            new_diffs_mask = 0
            conflict = False
            for a in row:
                d = x - a
                if d > max_diff:
                    conflict = True
                    break
                bit = 1 << d
                if new_diffs_mask & bit:
                    conflict = True
                    break
                if used_mask & bit:
                    conflict = True
                    break
                new_diffs_mask |= bit
            
            if conflict:
                continue
            
            row.append(x)
            new_used = used_mask | new_diffs_mask
            
            result = _build_row(row, row_idx, new_used, rows, max_mark)
            if result is not None:
                return result
            row.pop()
        
        return None
    
    # Try scopes from 112 downward
    best_result = None
    for scope in [112, 111, 110, 109, 108, 107, 106, 105]:
        if time.time() > deadline - 5:
            break
        
        result = _backtrack(0, 0, [], scope)
        if result is not None:
            if _verify_solution(result, n_rows, k):
                best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    return _greedy_construct(n_rows, k)
