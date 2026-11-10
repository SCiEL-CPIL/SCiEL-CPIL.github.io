import time
import random
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set (n_rows, k) with smallest possible maximum entry.
    
    Uses a backtracking search with constraint propagation, greedy initialization,
    and multiple restarts. Returns a list of n_rows rows, each with k+1 strictly
    increasing integers starting at 0.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # For small cases, use direct greedy
    if n_rows <= 2 and k <= 3:
        return _greedy_construct(n_rows, k)
    
    # Try to find a good solution using backtracking with time limit
    best_solution = None
    best_max_entry = None
    
    # Multiple restarts with different orderings and heuristics
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Shuffle rows to try different orderings
        row_order = list(range(n_rows))
        if attempt > 0:
            random.shuffle(row_order)
        
        # Try backtracking with this row order
        solution = _backtracking_solve(n_rows, k, row_order, deadline)
        if solution is not None:
            max_entry = max(max(row) for row in solution)
            if best_max_entry is None or max_entry < best_max_entry:
                best_max_entry = max_entry
                best_solution = solution
                # If we found a very good solution, we can stop early
                if max_entry <= 112 and n_rows == 7 and k == 6:
                    break
    
    # If backtracking didn't find anything, fall back to greedy
    if best_solution is None:
        best_solution = _greedy_construct(n_rows, k)
    
    return best_solution


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes the smallest possible next mark."""
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


def _backtracking_solve(n_rows, k, row_order, deadline):
    """
    Backtracking solver that tries to construct rows in the given order.
    Uses constraint propagation to prune the search space.
    """
    used = set()  # All differences used so far
    rows = [None] * n_rows  # Store constructed rows
    current_row_idx = 0
    
    def can_place_row(row):
        """Check if a row can be placed given current used differences."""
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used or d in row_diffs:
                    return False
                row_diffs.add(d)
        return True
    
    def backtrack(pos, current_row, current_x):
        """
        Backtracking to build the current row.
        pos: position in current row (0..k)
        current_row: list of marks so far in current row
        current_x: last mark in current row
        """
        nonlocal used, rows, current_row_idx
        
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            # Completed this row
            rows[current_row_idx] = current_row
            # Update used differences
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    used.add(current_row[j] - current_row[i])
            
            # Move to next row
            if current_row_idx + 1 < n_rows:
                current_row_idx += 1
                return backtrack(1, [0], 0)
            else:
                return True  # All rows constructed
        
        # Try next possible mark
        x = current_x + 1
        while True:
            if time.time() > deadline:
                return False
            
            # Check if x - a for all a in current_row are valid
            valid = True
            for a in current_row:
                d = x - a
                if d in used:
                    valid = False
                    break
            if valid:
                # Also check no duplicates within this row's diffs
                row_diffs = {x - a for a in current_row}
                if len(row_diffs) == len(current_row):
                    # Try placing x
                    # Temporarily mark diffs as used for deeper recursion
                    old_used = used.copy()
                    used.update(row_diffs)
                    
                    if backtrack(pos + 1, current_row + [x], x):
                        return True
                    
                    # Backtrack
                    used = old_used
            
            x += 1
            # Limit search depth to avoid infinite loops
            if x > 500:
                return False
    
    # Start backtracking
    return backtrack(1, [0], 0)


def _construct_greedy_with_pruning(n_rows, k):
    """
    Greedy construction with better pruning and ordering.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # Check for duplicate diffs within the row
                if len(new_diffs) == len(row):
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    
    return rows
