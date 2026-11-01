import time
import random
import numpy as np

_DEADLINE = 120.0


def _greedy_rows(n_rows, k):
    """Quick greedy construction to get an initial upper bound."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = [x - a for a in row]
            if not any(d in used for d in new_diffs):
                row.append(x)
                for d in new_diffs:
                    used.add(d)
        rows.append(row)
    return rows


def _scope(rows):
    m = 0
    for r in rows:
        if r:
            m = max(m, r[-1])
    return m


def _backtrack_solve(n_rows, k, M, deadline):
    """
    Backtracking with forward checking to find a valid solution with scope <= M.
    Returns (True, rows) if found, (False, None) if not.
    Uses diff_counts for fast conflict checking.
    """
    timed_out = [False]
    
    # Maximum difference we need to track
    max_diff = M
    
    # diff_counts[d] = number of times difference d is used
    diff_counts = [0] * (max_diff + 1)
    
    def get_row_diffs(row):
        """Get all positive differences in a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= max_diff:
                    diffs.append(d)
        return diffs
    
    def add_row_diffs(row):
        for d in get_row_diffs(row):
            diff_counts[d] += 1
    
    def remove_row_diffs(row):
        for d in get_row_diffs(row):
            diff_counts[d] -= 1
    
    def is_valid_placement(row, x):
        """Check if placing x at end of row creates no conflicts."""
        for a in row:
            d = x - a
            if d > max_diff:
                return False
            if diff_counts[d] > 0:
                return False
        return True
    
    def try_build_row(row_idx, current_row, used_in_row):
        """
        Try to build row row_idx. current_row is the partial row being built.
        used_in_row is a set of diffs already used within this row (for internal consistency).
        Returns (True, completed_row) or (False, None).
        """
        if timed_out[0]:
            return False, None
        
        if time.time() > deadline:
            timed_out[0] = True
            return False, None
        
        if len(current_row) == k + 1:
            # Row complete
            add_row_diffs(current_row)
            return True, current_row[:]
        
        last = current_row[-1]
        
        # Forward checking: the minimum gap needed
        # We need to place (k+1 - len(current_row)) more marks
        # The minimum possible span for remaining marks is at least
        # sum of smallest (k+1-len) gaps, but we just need x <= M
        
        # Try candidates from last+1 to M
        # Optimization: we can skip some candidates
        start_x = last + 1
        end_x = M
        
        # For efficiency, try a limited number of candidates
        # Focus on values that leave room for remaining marks
        remaining = k + 1 - len(current_row)
        
        # The smallest possible next value that allows completing the row
        # If we need 'remaining' more marks, and the minimum gap is 1,
        # we need at least 'remaining' more units of space
        # So x should be at most M - (remaining - 1) to leave room
        max_useful_x = M - (remaining - 1)
        
        if max_useful_x < start_x:
            return False, None
        
        end_x = min(end_x, max_useful_x)
        
        # Try candidates
        for x in range(start_x, end_x + 1):
            if timed_out[0]:
                return False, None
            if time.time() > deadline:
                timed_out[0] = True
                return False, None
            
            # Check if x is valid
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                if d > max_diff or diff_counts[d] > 0:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check internal consistency (diffs within the row must be distinct)
            # Actually, the problem says all differences across ALL rows must be distinct,
            # so internal consistency is automatically handled by the global diff_counts
            # But we also need to ensure that within this row, we don't create duplicate diffs
            # Wait - the diff_counts already tracks global usage, so if a diff is used in this row,
            # it will be in diff_counts. But we're checking before adding, so we need to make sure
            # that the new diffs don't conflict with each other within this row either.
            
            # Check for duplicate diffs within the new placement
            if len(set(new_diffs)) != len(new_diffs):
                continue
            
            # Tentatively place x
            current_row.append(x)
            for d in new_diffs:
                diff_counts[d] += 1
            
            # Recurse
            res, _ = try_build_row(row_idx, current_row, None)
            
            if res:
                return True, current_row[:]
            
            # Undo
            for d in new_diffs:
                diff_counts[d] -= 1
            current_row.pop()
        
        return False, None
    
    # Main backtracking over rows
    def backtrack(row_idx, completed_rows):
        if timed_out[0]:
            return False, None
        
        if time.time() > deadline:
            timed_out[0] = True
            return False, None
        
        if row_idx == n_rows:
            return True, completed_rows[:]
        
        # Try to build row row_idx
        res, new_row = try_build_row(row_idx, [0], None)
        
        if res:
            completed_rows.append(new_row)
            res2, sol = backtrack(row_idx + 1, completed_rows)
            if res2:
                return True, sol
            completed_rows.pop()
        
        return False, None
    
    return backtrack(0, [])


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _DEADLINE - 10.0  # safety margin
    
    # 1) Greedy baseline
    best_rows = _greedy_rows(n_rows, k)
    best_scope = _scope(best_rows)
    
    # 2) Lower bound: total number of differences
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # If best_scope is already at the lower bound, we're optimal
    if best_scope <= total_diffs:
        return best_rows
    
    # 3) Binary search on scope
    L = total_diffs
    U = best_scope - 1
    
    # We'll try to find the minimal M where a solution exists
    # Use backtracking with a time budget per attempt
    
    # Allocate time: use 70% of remaining time for binary search
    remaining_time = deadline - t0
    search_time = remaining_time * 0.7
    
    # Binary search
    while L <= U and not timed_out_global(t0, deadline - search_time):
        M = (L + U) // 2
        
        # Give each backtracking attempt a time budget
        attempt_deadline = min(deadline - 2.0, t0 + search_time)
        
        found, sol = _backtrack_solve(n_rows, k, M, attempt_deadline)
        
        if found:
            best_rows = sol
            best_scope = M
            U = M - 1
        else:
            L = M + 1
    
    return best_rows


def timed_out_global(t0, deadline):
    return time.time() > deadline
