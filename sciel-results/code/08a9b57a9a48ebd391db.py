"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _check_valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _greedy_rows(n_rows, k, used=None):
    """Greedy construction of n_rows rows."""
    if used is None:
        used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows, used


def _try_replace_row(rows, row_idx, k, max_allowed, deadline):
    """Try to find a replacement for rows[row_idx] with max entry <= max_allowed.
    Returns the new row if found, else None."""
    other_rows = rows[:row_idx] + rows[row_idx + 1:]
    other_used = set()
    for r in other_rows:
        other_used |= _row_diffs(r)
    
    # Backtrack to find a valid row
    row = [0]
    row_diffs = set()
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        for x in range(row[-1] + 1, max_allowed + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & other_used or new_diffs & row_diffs:
                continue
            row.append(x)
            row_diffs.update(new_diffs)
            if backtrack(pos + 1):
                return True
            row_diffs.difference_update(new_diffs)
            row.pop()
        return False
    
    if backtrack(1):
        return list(row)
    return None


def _local_search(rows, k, deadline):
    """Iteratively try to replace each row with one that has a smaller max entry."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)
    no_improve_count = 0
    
    while time.time() < deadline - 3.0:
        improved = False
        
        # Try each row in order of largest max first
        order = sorted(range(len(best)), key=lambda i: -max(best[i]))
        
        for row_idx in order:
            if time.time() > deadline - 3.0:
                break
            current_max = max(best[row_idx])
            if current_max <= best_scope - 1:
                continue
            
            # Try to find a row with max < current_max
            for target_max in range(current_max - 1, max(current_max - 10, 0), -1):
                if time.time() > deadline - 3.0:
                    break
                new_row = _try_replace_row(best, row_idx, k, target_max, deadline)
                if new_row is not None:
                    best[row_idx] = new_row
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        # Revert if it didn't improve the global scope
                        pass  # Keep the new row anyway, it might help later
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                # Try random perturbation
                ridx = random.randint(0, len(best) - 1)
                old_row = best[ridx]
                # Try to find any valid replacement with same or slightly smaller max
                new_row = _try_replace_row(best, ridx, k, max(old_row), deadline)
                if new_row is not None and new_row != old_row:
                    best[ridx] = new_row
                    no_improve_count = 0
                else:
                    break
        else:
            no_improve_count = 0
    
    return best, best_scope


def construct(n_rows, k):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    best_rows = None
    best_scope = float('inf')
    
    # Greedy baseline
    greedy_rows, greedy_used = _greedy_rows(n_rows, k)
    best_rows = greedy_rows
    best_scope = max(max(r) for r in best_rows)
    
    # Local search to improve
    if time.time() < deadline - 5.0:
        local_rows, local_scope = _local_search(best_rows, k, deadline)
        if local_scope < best_scope:
            best_rows = local_rows
            best_scope = local_scope
    
    # Try targeted backtracking: fix first n_rows-1 rows, search last row
    if n_rows >= 1 and time.time() < deadline - 5.0:
        for n_fixed in range(n_rows - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            fixed = best_rows[:n_fixed]
            fixed_used = set()
            for r in fixed:
                fixed_used |= _row_diffs(r)
            
            remaining = n_rows - n_fixed
            # Search for remaining rows with decreasing max
            current_max = best_scope
            for target in range(current_max - 1, max(current_max - 30, 0), -1):
                if time.time() > deadline - 5.0:
                    break
                found = []
                used_so_far = set(fixed_used)
                success = True
                for _ in range(remaining):
                    new_row = _try_replace_row(fixed + found, len(fixed) + len(found), k, target, deadline)
                    if new_row is None:
                        success = False
                        break
                    found.append(new_row)
                    used_so_far |= _row_diffs(new_row)
                
                if success:
                    candidate = fixed + found
                    scope = max(max(r) for r in candidate)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = candidate
                    break
    
    # Final local search pass
    if time.time() < deadline - 3.0:
        local_rows, local_scope = _local_search(best_rows, k, deadline)
        if local_scope < best_scope:
            best_rows = local_rows
            best_scope = local_scope
    
    return best_rows
# EVOLVE-BLOCK-END
