import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
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

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _solve_bitmask(n_rows, k, max_entry, time_limit=2.0):
    """
    Backtracking search using bitmasks for speed.
    """
    start_time = time.time()
    used_mask = 0
    rows = []
    
    # Helper to check if a set of diffs conflicts with used_mask
    # diffs is a list of ints
    def has_conflict(diffs):
        mask = 0
        for d in diffs:
            if d > max_entry:
                return True # Should not happen if we constrain x
            mask |= (1 << d)
        return (used_mask & mask) != 0

    def backtrack(row_idx, current_row, last_val):
        nonlocal used_mask, rows
        
        if time.time() - start_time > time_limit:
            return False
            
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row[:])
            # Save state
            old_used = used_mask
            # Mark diffs of current_row
            mask = 0
            for i in range(len(current_row)):
                for j in range(i):
                    d = current_row[i] - current_row[j]
                    mask |= (1 << d)
            used_mask |= mask
            
            if backtrack(row_idx + 1, [0], 0):
                return True
            
            # Restore
            used_mask = old_used
            rows.pop()
            return False
        
        # Generate candidates for next value
        candidates = []
        # Heuristic: Start from last_val + 1
        # We can randomize the order
        start_x = last_val + 1
        # To allow randomization, we can pick a random subset or shuffle a range
        # But range is up to 112. Small enough to shuffle?
        # 112 items. Yes.
        
        # Optimization: Only consider x such that there is space for remaining elements?
        # Remaining elements: k + 1 - len(current_row)
        # Min gap is 1. So x + remaining <= max_entry.
        remaining = k + 1 - len(current_row)
        max_x_allowed = max_entry - (remaining - 1)
        
        if start_x > max_x_allowed:
            return False

        # Create list of candidates
        # We can iterate and collect valid ones
        valid_candidates = []
        for x in range(start_x, max_x_allowed + 1):
            # Check conflict with current_row
            conflict = False
            for y in current_row:
                d = x - y
                if d > max_entry:
                    conflict = True
                    break
                if (used_mask >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return False
            
        # Shuffle to allow different paths
        random.shuffle(valid_candidates)
        
        for x in valid_candidates:
            current_row.append(x)
            if backtrack(row_idx, current_row, x):
                return True
            current_row.pop()
            
        return False

    if backtrack(0, [0], 0):
        return rows
    return None

def construct(n_rows, k):
    try:
        start_time = time.time()
        deadline = start_time + 140.0  # Safety margin
        
        # 1. Greedy init
        best_rows = _greedy_init(n_rows, k)
        best_val = _global_max(best_rows)
        
        # 2. Try to improve
        # We try to find a solution with max_entry < best_val
        # We use the bitmask solver
        
        current_m = best_val - 1
        
        # If best_val is small, we can try more aggressively
        search_deadline = deadline - 5.0
        
        # Try decreasing M
        # We will attempt to solve for M = current_m, current_m - 1, etc.
        # But if M is infeasible, we stop.
        
        attempts_for_m = 0
        max_attempts_for_m = 3
        
        while current_m >= 50 and time.time() < search_deadline:
            rows_found = _solve_bitmask(n_rows, k, current_m, time_limit=1.5)
            
            if rows_found:
                actual_max = _global_max(rows_found)
                if actual_max < best_val:
                    best_val = actual_max
                    best_rows = rows_found
                    current_m = actual_max - 1
                    attempts_for_m = 0
                else:
                    # Found solution but not better? Should not happen if current_m < best_val
                    break
            else:
                attempts_for_m += 1
                if attempts_for_m >= max_attempts_for_m:
                    # Assume infeasible, stop
                    break
                # Otherwise, retry with randomization (handled inside solver)
                
    except Exception:
        # If anything goes wrong, return the greedy solution
        # Ensure best_rows is defined
        if 'best_rows' not in locals() or best_rows is None:
             best_rows = _greedy_init(n_rows, k)
             
    return best_rows

# EVOLVE-BLOCK-END
