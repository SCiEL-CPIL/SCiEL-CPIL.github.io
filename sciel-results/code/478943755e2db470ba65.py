"""Improved program for the difference triangle set task.

Strategy:
- Start with the fast greedy construction to get a valid baseline.
- Attempt local search to reduce the maximum entry by trying to shift individual
  marks to smaller unused positions that preserve distinctness.
- Use a time budget to ensure we return within the limit.
"""
import time
import random
import sys

def _greedy_construct(n_rows, k):
    """Original greedy construction: fast baseline."""
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

def _validate(rows, k):
    """Check that all positive differences within rows are distinct across all rows."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _max_entry(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _try_improve(rows, k, used, deadline, rng):
    """
    Local search: try to move marks to smaller positions to reduce the max entry.
    We process marks in order of largest value first, trying to find a smaller
    position that keeps all differences distinct.
    """
    rows = [row[:] for row in rows]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Collect all (row_idx, mark_idx, value) sorted by value descending
        entries = []
        for ri, row in enumerate(rows):
            for mi in range(1, len(row)):  # skip index 0 (always 0)
                entries.append((row[mi], ri, mi))
        entries.sort(reverse=True)
        
        for val, ri, mi in entries:
            if time.time() >= deadline:
                break
            row = rows[ri]
            # Try to find a smaller position for row[mi]
            # The new position must be > row[mi-1] and < val
            # And all new differences must not conflict
            best_new = None
            # Try positions from row[mi-1]+1 up to val-1
            # But we want to minimize, so try smaller first
            # However, we need to ensure the row remains increasing
            for new_val in range(row[mi - 1] + 1, val):
                # Check if moving row[mi] to new_val works
                # Remove old differences involving row[mi]
                old_diffs = set()
                for j in range(mi):
                    old_diffs.add(val - row[j])
                for j in range(mi + 1, len(row)):
                    old_diffs.add(row[j] - val)
                
                # Temporarily remove old diffs from used
                used_temp = used.copy()
                for d in old_diffs:
                    used_temp.discard(d)
                
                # Check new differences
                new_diffs = set()
                for j in range(mi):
                    new_diffs.add(new_val - row[j])
                for j in range(mi + 1, len(row)):
                    new_diffs.add(row[j] - new_val)
                
                # Check if any new diff is already used
                if new_diffs & used_temp:
                    continue
                
                # Check if new diffs are all distinct (they should be since row is increasing)
                # Also need to check that new_val > row[mi-1] which we ensured
                # And new_val < row[mi+1] if mi+1 < len(row)
                if mi + 1 < len(row) and new_val >= row[mi + 1]:
                    continue
                
                # This move is valid
                best_new = new_val
                break
            
            if best_new is not None:
                # Apply the move
                row[mi] = best_new
                # Update used
                old_diffs = set()
                for j in range(mi):
                    old_diffs.add(val - row[j])
                for j in range(mi + 1, len(row)):
                    old_diffs.add(row[j] - val)
                for d in old_diffs:
                    used.discard(d)
                
                new_diffs = set()
                for j in range(mi):
                    new_diffs.add(best_new - row[j])
                for j in range(mi + 1, len(row)):
                    new_diffs.add(row[j] - best_new)
                used |= new_diffs
                
                improved = True
                # Re-sort entries since values changed
                entries = []
                for ri2, row2 in enumerate(rows):
                    for mi2 in range(1, len(row2)):
                        entries.append((row2[mi2], ri2, mi2))
                entries.sort(reverse=True)
                break  # Restart the loop with updated entries
    
    return rows

def _try_rebuild_last_row(rows, k, used, deadline, rng):
    """
    Try to rebuild the last row with a smaller maximum entry.
    """
    if not rows:
        return rows
    
    rows = [row[:] for row in rows]
    n_rows = len(rows)
    
    # Remove last row from used
    last_row = rows[-1]
    for i in range(len(last_row)):
        for j in range(i + 1, len(last_row)):
            used.discard(last_row[j] - last_row[i])
    
    # Try to construct a new last row with smaller max
    best_row = last_row
    best_max = max(last_row)
    
    # Try greedy with a cap
    for cap in range(best_max - 1, 0, -1):
        if time.time() >= deadline:
            break
        
        row, x = [0], 0
        success = True
        while len(row) < k + 1:
            x += 1
            if x > cap:
                success = False
                break
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                # Temporarily add to used for this row's construction
                # But we need to be careful: we're building one row at a time
                # For simplicity, let's just check against the global used set
                # and the row's own diffs
        if success:
            # Verify
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in row_diffs:
                        success = False
                        break
                    row_diffs.add(d)
            if success:
                best_row = row
                best_max = max(row)
                break
    
    rows[-1] = best_row
    # Update used
    for i in range(len(best_row)):
        for j in range(i + 1, len(best_row)):
            used.add(best_row[j] - best_row[i])
    
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Step 1: Greedy baseline
    rows = _greedy_construct(n_rows, k)
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    initial_max = _max_entry(rows)
    
    # Step 2: Local search to improve
    if time.time() < deadline - 10:
        rows = _try_improve(rows, k, used, deadline - 10, random.Random(42))
        # Rebuild used set to be safe
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
    
    # Step 3: Try to rebuild last row with smaller max
    if time.time() < deadline - 5:
        rows = _try_rebuild_last_row(rows, k, used, deadline - 5, random.Random(42))
    
    # Final validation
    if not _validate(rows, k):
        # Fallback to greedy
        rows = _greedy_construct(n_rows, k)
    
    return rows
