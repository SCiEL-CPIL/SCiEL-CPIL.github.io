"""
Structural change: replace the naive greedy construction with a randomized
local-search / simulated-annealing style optimizer over the set of rows.

Key ideas:
1. Start from a greedy construction (same as seed) to get a valid baseline.
2. Define the objective as the maximum entry (scope) across all rows.
3. Use a randomized perturbation + local improvement loop:
   - Pick a random row and a random position within it.
   - Try to reduce that entry by searching for a smaller value that keeps
     the row strictly increasing and all differences globally unique.
   - If a smaller value works, accept it (this strictly reduces or keeps
     the scope).
   - Also try "swap" moves: pick two rows and attempt to exchange some
     entries to reduce the overall max.
4. Track the best solution found (by max entry) and return it before the
   deadline.

This is a different optimization family (local search with a specific
objective = max entry) rather than pure greedy construction.
"""
import time
import random
import sys

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return the union of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used

def _is_valid(rows):
    """Check that all rows are valid and all differences are globally unique."""
    used = set()
    for row in rows:
        if len(row) != len(set(row)):
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[j] <= row[i]:
                    return False
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, rng):
    """Greedy construction with optional randomization in tie-breaking."""
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

def _try_reduce_entry(rows, row_idx, pos_idx, used):
    """
    Try to reduce rows[row_idx][pos_idx] to a smaller value.
    Returns (new_row, new_used) if successful, else None.
    """
    row = rows[row_idx]
    current_val = row[pos_idx]
    
    # Determine bounds for the new value
    lower = row[pos_idx - 1] + 1 if pos_idx > 0 else 1
    upper = current_val - 1  # we want strictly smaller
    
    if upper < lower:
        return None
    
    # Compute diffs that involve position pos_idx in the current row
    # These are the diffs that will change
    old_diffs = set()
    for i in range(len(row)):
        if i != pos_idx:
            if row[pos_idx] > row[i]:
                old_diffs.add(row[pos_idx] - row[i])
            else:
                old_diffs.add(row[i] - row[pos_idx])
    
    # Try values from upper down to lower (prefer smaller)
    for new_val in range(upper, lower - 1, -1):
        # Check strictly increasing
        if pos_idx > 0 and new_val <= row[pos_idx - 1]:
            continue
        if pos_idx < len(row) - 1 and new_val >= row[pos_idx + 1]:
            continue
        
        # Compute new diffs involving pos_idx
        new_diffs = set()
        for i in range(len(row)):
            if i != pos_idx:
                if new_val > row[i]:
                    new_diffs.add(new_val - row[i])
                else:
                    new_diffs.add(row[i] - new_val)
        
        # The diffs that are removed are old_diffs not in new_diffs
        # The diffs that are added are new_diffs not in old_diffs
        removed = old_diffs - new_diffs
        added = new_diffs - old_diffs
        
        # Check that added diffs are not already used (outside this row's other diffs)
        # We need to check against used minus the old_diffs of this row (since those will be freed)
        # Actually, used contains ALL diffs from ALL rows. The old_diffs from this row are in used.
        # After the change, the new_diffs replace old_diffs for this row.
        # So we need: added & (used - old_diffs) == empty
        # Because old_diffs are being removed from used, and other rows' diffs remain.
        
        if added & (used - old_diffs):
            continue
        
        # Also need to make sure the new row is still strictly increasing
        # (already checked above for neighbors, but let's verify full row)
        new_row = row[:pos_idx] + [new_val] + row[pos_idx + 1:]
        for i in range(len(new_row) - 1):
            if new_row[i + 1] <= new_row[i]:
                break
        else:
            # Valid! Update used
            new_used = used - old_diffs | new_diffs
            return new_row, new_used
    
    return None

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Local search to reduce the max entry.
    Returns the best rows found.
    """
    best_rows = [row[:] for row in rows]
    best_max = _max_entry(best_rows)
    best_used = _all_diffs(best_rows)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5.0:
        # Pick a random row
        row_idx = rng.randint(0, n_rows - 1)
        row = rows[row_idx]
        
        # Pick a random position (not 0, since it's always 0)
        if len(row) <= 1:
            continue
        pos_idx = rng.randint(1, len(row) - 1)
        
        result = _try_reduce_entry(rows, row_idx, pos_idx, best_used)
        if result is not None:
            new_row, new_used = result
            rows[row_idx] = new_row
            best_used = new_used
            
            new_max = _max_entry(rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart with a fresh greedy to escape local optima
            rows = _greedy_construct(n_rows, k, rng)
            best_used = _all_diffs(rows)
            new_max = _max_entry(rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
            no_improve_count = 0
    
    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 115.0  # 115s to leave margin before 120s limit
    
    rng = random.Random(42)
    
    # Initial greedy construction
    rows = _greedy_construct(n_rows, k, rng)
    if not _is_valid(rows):
        # Fallback: return the greedy result even if somehow invalid
        return rows
    
    # Local search to improve
    improved_rows = _local_search(rows, n_rows, k, deadline, rng)
    
    # Final validity check
    if _is_valid(improved_rows):
        return improved_rows
    else:
        return rows
