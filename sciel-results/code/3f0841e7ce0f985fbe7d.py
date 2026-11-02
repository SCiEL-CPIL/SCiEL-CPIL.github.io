"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _get_scope(rows):
    """Return the largest entry in any row."""
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_init(n_rows, k):
    """Greedy initialization: build rows one by one with smallest possible marks."""
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

def _rebuild_row(row, used, k, max_val):
    """
    Try to rebuild a row with entries in [0, max_val] that avoids 'used' differences.
    Returns a new valid row or None.
    Uses backtracking with pruning.
    """
    n = k + 1  # number of elements
    new_row = [0]
    row_diffs = set()
    
    def backtrack(pos, current_max):
        if pos == n:
            return True
        start = current_max + 1
        end = min(max_val, current_max + max(1, (max_val - current_max) // (n - pos)))
        # We need to leave room for remaining elements
        # Remaining elements: n - pos - 1
        # Each needs at least 1 more than previous
        # So max_val - current_max >= n - pos - 1
        if max_val - current_max < (n - pos - 1):
            return False
        for v in range(start, end + 1):
            # Check new diffs
            new_diffs = set()
            ok = True
            for i in range(pos):
                d = v - new_row[i]
                if d in used or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            new_row.append(v)
            if backtrack(pos + 1, v):
                return True
            new_row.pop()
        return False
    
    if backtrack(1, 0):
        # Compute all diffs for this row
        final_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                final_diffs.add(new_row[j] - new_row[i])
        return new_row, final_diffs
    return None, None

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _optimize(rows, n_rows, k, deadline):
    """
    Multi-strategy optimization:
    1. Row reconstruction: rebuild individual rows to fit in smaller scope
    2. Row swaps: try swapping rows to improve packing
    3. Systematic scope reduction: try to reduce max entry
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build used set
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    
    current_scope = _get_scope(rows)
    
    # Strategy 1: Try to reduce scope by rebuilding rows
    # Sort rows by their max entry (largest first)
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Try to reduce the scope by 1
        target_scope = best_scope - 1
        if target_scope < 0:
            break
        
        # Find which row has the max entry
        max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        
        if rows[max_row_idx][-1] < target_scope:
            # No row exceeds target_scope, but maybe we can reduce further
            # Actually, if no row has entry >= target_scope, scope is already < target_scope
            # This shouldn't happen since best_scope is the max
            pass
        
        # Try to rebuild the row with the largest entry to fit in target_scope
        old_row = rows[max_row_idx]
        old_diffs = _row_diffs(old_row)
        remaining_used = used - old_diffs
        
        # Rebuild this row
        result = _rebuild_row(old_row, remaining_used, k, target_scope)
        if result[0] is not None:
            new_row, new_diffs = result
            # Check if new_diffs conflicts with remaining_used
            if not (new_diffs & remaining_used):
                rows[max_row_idx] = new_row
                used = remaining_used | new_diffs
                new_scope = _get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
        
        # Also try rebuilding other rows randomly
        if random.random() < 0.5:
            r_idx = random.randint(0, n_rows - 1)
            if r_idx == max_row_idx:
                continue
            old_row = rows[r_idx]
            old_diffs = _row_diffs(old_row)
            remaining_used = used - old_diffs
            # Try to fit in current best_scope
            result = _rebuild_row(old_row, remaining_used, k, best_scope)
            if result[0] is not None:
                new_row, new_diffs = result
                if not (new_diffs & remaining_used):
                    rows[r_idx] = new_row
                    used = remaining_used | new_diffs
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
    
    # Strategy 2: Local search - single element moves with validity check
    T = 30.0
    cooling = 0.998
    min_T = 0.5
    
    while T > min_T and time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]
        
        max_val = best_scope
        if random.random() < 0.6:
            new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
        else:
            new_val = random.randint(1, max_val)
        
        if new_val == old_val:
            T *= cooling
            continue
        
        if pos > 0 and new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            T *= cooling
            continue
        
        # Compute old and new diffs
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - old_val)
        
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)
        
        if new_diffs & (used - old_diffs):
            T *= cooling
            continue
        
        # Accept
        row[pos] = new_val
        used = (used - old_diffs) | new_diffs
        new_scope = _get_scope(rows)
        delta = new_scope - best_scope
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        else:
            row[pos] = old_val
            used = (used - new_diffs) | old_diffs
        
        T *= cooling
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with greedy solution
    rows = _greedy_init(n_rows, k)
    
    # Verify validity
    if not _is_valid(rows):
        return rows
    
    # Optimize
    rows = _optimize(rows, n_rows, k, deadline)
    
    # Final validation
    if not _is_valid(rows):
        return _greedy_init(n_rows, k)
    
    return rows
# EVOLVE-BLOCK-END
