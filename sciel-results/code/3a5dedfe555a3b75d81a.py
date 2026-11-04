"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _valid(rows):
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

def _scope(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _try_shrink(rows, used_diffs):
    """
    Attempt to shrink every element in every row as much as possible.
    Returns True if any change was made, False otherwise.
    Modifies rows and used_diffs in place.
    """
    changed = False
    # We iterate multiple times because shrinking one element might allow shrinking another
    # We iterate rows in reverse order of their max value to attack the largest scopes first
    row_indices = sorted(range(len(rows)), key=lambda i: max(rows[i]), reverse=True)
    
    for r_idx in row_indices:
        row = rows[r_idx]
        # Iterate elements from right to left
        for pos in range(len(row) - 1, 0, -1):
            old_val = row[pos]
            # The new value must be > row[pos-1]
            min_val = row[pos - 1] + 1
            
            # Try to reduce row[pos] to min_val, min_val+1, ... old_val-1
            for new_val in range(min_val, old_val):
                # Calculate differences that would be added/removed
                # Removed diffs: {old_val - row[i] for i in range(pos)}
                # Added diffs: {new_val - row[i] for i in range(pos)}
                
                removed = set()
                added = set()
                for i in range(pos):
                    removed.add(old_val - row[i])
                    added.add(new_val - row[i])
                
                # Check if 'added' conflicts with 'used_diffs' excluding 'removed'
                # We must ensure that any difference in 'added' is not already in 'used_diffs' 
                # unless it is in 'removed' (which is being replaced).
                # Actually, since we are changing the value, the 'removed' diffs are no longer present.
                # So we check if (added - removed) intersects with (used_diffs - removed).
                # Simpler: 
                # 1. Remove 'removed' from used_diffs
                # 2. Check if 'added' intersects with new used_diffs
                # 3. If not, add 'added' to used_diffs and accept.
                
                # Tentatively remove
                used_diffs.difference_update(removed)
                
                if not (added & used_diffs):
                    # Also check internal consistency within the row?
                    # The 'added' diffs are new differences involving pos.
                    # Do they conflict with other diffs in the row that do NOT involve pos?
                    # Yes, we need to check against the rest of the row's diffs.
                    # However, 'used_diffs' currently contains diffs from ALL other rows 
                    # AND the current row's diffs EXCEPT those involving pos (since we removed them).
                    # Wait, used_diffs contains ALL diffs from all rows.
                    # When we remove 'removed', we have removed the diffs involving pos from the current row.
                    # The remaining diffs in the current row (not involving pos) are still in used_diffs.
                    # So checking (added & used_diffs) covers both global conflicts and internal row conflicts 
                    # (with elements < pos).
                    # What about internal conflicts with elements > pos?
                    # Those diffs do not involve pos, so they are in used_diffs.
                    # So the check is sufficient.
                    
                    used_diffs.update(added)
                    row[pos] = new_val
                    changed = True
                    # If we changed, we restart the shrinking for this row? 
                    # Or just continue to the left? 
                    # Continuing to the left is safe because changing row[pos] 
                    # only affects diffs with elements < pos. 
                    # It does not affect the validity of elements > pos 
                    # (their diffs with pos are now different, but we just verified 
                    # they don't conflict with anything else).
                    # However, reducing row[pos] might allow row[pos+1] to be reduced?
                    # No, row[pos+1] > row[pos]. If we reduce row[pos], the gap to row[pos+1] increases.
                    # This doesn't help reduce row[pos+1].
                    # But reducing row[pos] might allow row[pos-1] to be reduced?
                    # No, row[pos-1] < row[pos].
                    # So we can just continue to pos-1.
                else:
                    # Revert
                    used_diffs.update(removed)
                    break # Try next smaller new_val? No, we iterate upwards from min_val.
                    # If min_val fails, larger values might work?
                    # Yes. We want the SMALLEST valid value.
                    # So we keep trying new_val = min_val, min_val+1, ...
                    # The loop continues.
            # End for new_val
        # End for pos
    # End for r_idx
    
    return changed

def _construct_initial(n_rows, k, rng):
    """
    Construct an initial valid set using greedy randomized search.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try random steps
            found = False
            # Estimate a reasonable max step
            # As we add more elements, steps need to be larger to avoid collisions
            max_step = 50 
            for _ in range(100):
                step = rng.randint(1, max_step)
                x = row[-1] + step
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    found = True
                    break
            if not found:
                # Fallback: linear search
                x = row[-1]
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        break
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    
    return rows

def _shake(rows, used_diffs, rng, intensity=0.2):
    """
    Randomly perturb the solution to escape local minima.
    Returns True if the solution is still valid after shaking.
    """
    # Pick a random row and a random element (not the first)
    r_idx = rng.randint(0, len(rows) - 1)
    row = rows[r_idx]
    pos = rng.randint(1, len(row) - 1)
    
    old_val = row[pos]
    # Try to move it to a random nearby value
    # Range: [row[pos-1]+1, old_val + 20]
    min_val = row[pos - 1] + 1
    max_val = old_val + 30
    
    for _ in range(50):
        new_val = rng.randint(min_val, max_val)
        if new_val == old_val:
            continue
            
        removed = set()
        added = set()
        for i in range(pos):
            removed.add(old_val - row[i])
            added.add(new_val - row[i])
            
        # Check validity
        used_diffs.difference_update(removed)
        if not (added & used_diffs):
            # Valid move
            used_diffs.update(added)
            row[pos] = new_val
            return True
        else:
            used_diffs.update(removed)
            
    return False

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate several initial candidates
    # We want diverse starting points
    candidates = []
    
    # Deterministic greedy (seed 0)
    rng0 = random.Random(0)
    rows0 = _construct_initial(n_rows, k, rng0)
    candidates.append(rows0)
    
    # Randomized greedy
    for seed in range(1, 20):
        if time.time() > deadline - 20.0:
            break
        rng = random.Random(seed)
        rows = _construct_initial(n_rows, k, rng)
        candidates.append(rows)
        
    # Phase 2: Local Search on each candidate
    for rows in candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Compute used diffs
        used_diffs = set()
        for r in rows:
            used_diffs.update(_row_diffs(r))
        
        # Optimization loop: Shake and Shrink
        current_scope = _scope(rows)
        no_improve_count = 0
        
        while time.time() < deadline - 5.0:
            # 1. Shrink
            if _try_shrink(rows, used_diffs):
                current_scope = _scope(rows)
                no_improve_count = 0
            else:
                # 2. Shake if stuck
                if no_improve_count > 5:
                    # Increase shake intensity if stuck for a long time
                    if _shake(rows, used_diffs, random.Random(int(time.time()*1000) % 10000)):
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                
                if no_improve_count > 50:
                    break # Give up on this candidate
                    
        # Update best
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [list(r) for r in rows]
            
    # Phase 3: Final aggressive shrink on the best solution
    if best_rows is not None and time.time() < deadline - 2.0:
        used_diffs = set()
        for r in best_rows:
            used_diffs.update(_row_diffs(r))
        
        # Do a few more passes of shrink
        for _ in range(10):
            if not _try_shrink(best_rows, used_diffs):
                break
        
        best_scope = _scope(best_rows)

    if best_rows is None:
        # Fallback: return a simple valid solution (should not happen)
        best_rows = _construct_initial(n_rows, k, random.Random(123))
        
    return best_rows

# EVOLVE-BLOCK-END
