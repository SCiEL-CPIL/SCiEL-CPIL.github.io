"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def _greedy_fallback(n_rows, k):
    """Greedy construction as fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _scope(rows):
    return max(max(row) for row in rows)

def _validate(rows):
    """Check if all differences across all rows are distinct."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _backtrack_row(row, used_diffs, max_val, k):
    """
    Backtrack to complete a single row given its current partial state.
    row: list of integers (starts with 0)
    used_diffs: set of all differences already used
    max_val: upper bound on entries
    k: number of intervals (so row should have k+1 entries)
    Returns True if row can be completed, modifying row in place.
    """
    if len(row) == k + 1:
        return True
    
    last = row[-1]
    for x in range(last + 1, max_val + 1):
        # Check if x creates any duplicate difference
        valid = True
        new_diffs = []
        for a in row:
            d = x - a
            if d in used_diffs:
                valid = False
                break
            new_diffs.append(d)
        
        if not valid:
            continue
        
        # Forward checking: verify that remaining positions can still be filled
        # Simple check: ensure we don't exceed max_val
        row.append(x)
        used_diffs.update(new_diffs)
        
        if _backtrack_row(row, used_diffs, max_val, k):
            return True
        
        # Undo
        for d in new_diffs:
            used_diffs.discard(d)
        row.pop()
    
    return False

def _construct_with_bound(n_rows, k, max_val, time_limit):
    """
    Try to construct n_rows rows of k+1 integers (starting at 0) with all
    entries <= max_val, using backtracking with forward checking.
    Returns list of rows or None if not found within time_limit.
    """
    start = time.time()
    used_diffs = set()
    rows = []
    
    def _build_row(row_idx):
        if time.time() - start > time_limit:
            return False
        
        if row_idx == n_rows:
            return True
        
        row = [0]
        if not _backtrack_row(row, used_diffs, max_val, k):
            return False
        
        rows.append(row)
        # The differences from this row are already in used_diffs
        
        if _build_row(row_idx + 1):
            return True
        
        # Backtrack: remove this row and its differences
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.discard(row[j] - row[i])
        rows.pop()
        return False
    
    if _build_row(0):
        return rows
    return None

def _local_search(rows, n_rows, k, time_limit):
    """
    Local search to reduce scope. Tries to move entries to smaller values
    while maintaining validity.
    """
    start = time.time()
    best = [row[:] for row in rows]
    best_scope = _scope(best)
    
    while time.time() - start < time_limit:
        improved = False
        # Try to reduce each entry
        for r in range(n_rows):
            row = best[r]
            for pos in range(len(row) - 1, 0, -1):  # from right to left
                old_val = row[pos]
                # Try to decrease
                for new_val in range(old_val - 1, row[pos-1], -1):
                    # Check if moving this entry to new_val keeps all diffs unique
                    # Compute all diffs with the new value
                    test_row = row[:]
                    test_row[pos] = new_val
                    
                    # Check ordering
                    if pos > 0 and new_val <= test_row[pos-1]:
                        break
                    if pos < len(test_row) - 1 and new_val >= test_row[pos+1]:
                        continue
                    
                    # Check all differences
                    all_diffs = set()
                    valid = True
                    for i in range(len(test_row)):
                        for j in range(i + 1, len(test_row)):
                            d = test_row[j] - test_row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check against other rows
                    for other_r in range(n_rows):
                        if other_r == r:
                            continue
                        other = best[other_r]
                        for i in range(len(other)):
                            for j in range(i + 1, len(other)):
                                d = other[j] - other[i]
                                if d in all_diffs:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        row[pos] = new_val
                        best_scope = _scope(best)
                        improved = True
                        break
        
        if not improved:
            break
    
    return best, best_scope

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # First, get a greedy solution as fallback
    greedy_rows = _greedy_fallback(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    
    # Try iterative deepening with backtracking
    # Start from a reasonable bound and increase
    # The counting bound is roughly: n_rows * k*(k+1)/2 differences needed
    # Each difference is in [1, max_val], so we need max_val >= n_rows * k*(k+1)/2
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs  # Simple lower bound
    
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Try to find a solution with scope close to lower bound
    for attempt_scope in range(lower_bound, greedy_scope + 1):
        if time.time() - start_time > deadline - 30:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 1:
            break
        
        # Use a fraction of remaining time for this attempt
        attempt_time = min(remaining_time * 0.3, 15)
        
        result = _construct_with_bound(n_rows, k, attempt_scope, attempt_time)
        if result is not None:
            scope = _scope(result)
            if scope < best_scope:
                best_rows = result
                best_scope = scope
            
            # If we found a solution at the lower bound, we can't do better
            if attempt_scope == lower_bound:
                break
            
            # Try local search to improve
            remaining_time = deadline - time.time() - 5
            if remaining_time > 5:
                improved_rows, improved_scope = _local_search(
                    result, n_rows, k, remaining_time * 0.5
                )
                if improved_scope < best_scope:
                    best_rows = improved_rows
                    best_scope = improved_scope
    
    # Final local search on best solution if time permits
    remaining_time = deadline - time.time() - 2
    if remaining_time > 3:
        improved_rows, improved_scope = _local_search(
            best_rows, n_rows, k, remaining_time
        )
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
    
    return best_rows
# EVOLVE-BLOCK-END
