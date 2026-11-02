import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, used_diffs=None):
    """
    Greedy construction of rows.
    If used_diffs is provided, it tries to avoid those differences (for perturbation).
    """
    rows = []
    current_used = set()
    if used_diffs is None:
        used_diffs = set()
    
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            # Check against internal row diffs and global used diffs
            if not new_diffs & current_used and not new_diffs & used_diffs:
                row.append(x)
                current_used |= new_diffs
            else:
                # If greedy gets stuck (shouldn't happen often if M is large enough),
                # we might need to backtracking or just force a larger jump.
                # For robustness in initialization, we just keep incrementing x.
                # However, if we are in a perturbation context, we might fail.
                # Let's add a small random jitter if stuck for a while? 
                # Actually, standard greedy usually works if M is sufficiently large.
                # If we are re-generating rows for perturbation, we pass in the *other* rows' diffs.
                pass
        
        rows.append(row)
        current_used |= {x - a for a in row[:-1] for x in [row[-1]]} # Just adding the last diffs is not enough, we need all.
        # Let's recompute properly for the set
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                row_diffs.add(row[j]-row[i])
        current_used |= row_diffs
        # We don't want to add this row's diffs to 'used_diffs' permanently for the next row 
        # if 'used_diffs' was meant to be the *external* constraint.
        # But in standard init, we want them to be disjoint.
        # In perturbation, we want the new rows to be disjoint from the *other* rows.
        # So 'used_diffs' should contain the diffs of the rows NOT being regenerated.
        
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _try_shrink_row(row, k, all_used_except_this_row):
    """
    Attempts to minimize the last element of the row by moving elements backwards.
    Returns the new row if successful in reducing the max, or the original row if not.
    """
    # We want to minimize row[k].
    # Strategy: Try to move row[k] down. If blocked, try to move row[k-1] down to free space?
    # This is complex. A simpler heuristic:
    # 1. Try to decrease row[k].
    # 2. If that fails, try to decrease row[k-1], which might allow row[k] to decrease?
    # Actually, let's just do a local search on the row itself, keeping the "external" constraints fixed.
    
    best_row = row[:]
    best_max = row[-1]
    
    # Try to shrink the last element
    # We iterate through positions from right to left? Or just try to reduce the max.
    
    # Simple hill climbing on the row
    improved = True
    while improved:
        improved = False
        for pi in range(k, 0, -1): # Start from the end
            old_val = best_row[pi]
            min_allowed = best_row[pi-1] + 1
            # Try to decrease
            for new_val in range(old_val - 1, min_allowed - 1, -1):
                if new_val <= min_allowed:
                    break
                # Check validity
                cand_row = best_row[:]
                cand_row[pi] = new_val
                
                # Check internal conflicts
                valid_internal = True
                new_diffs = set()
                # Diffs not involving pi
                for i in range(len(cand_row)):
                    if i == pi: continue
                    for j in range(i+1, len(cand_row)):
                        if j == pi: continue
                        new_diffs.add(cand_row[j] - cand_row[i])
                
                # Diffs involving pi
                for i in range(len(cand_row)):
                    if i == pi: continue
                    d = abs(cand_row[pi] - cand_row[i])
                    if d in new_diffs:
                        valid_internal = False
                        break
                    new_diffs.add(d)
                
                if not valid_internal:
                    continue
                
                # Check external conflicts
                # The new diffs must not be in all_used_except_this_row
                # Note: The old diffs of this row are NOT in all_used_except_this_row,
                # so we don't need to subtract them. We just check if any NEW diff (or diff that was in old row but is now different? No, we check all diffs of the candidate row)
                # Wait, if a diff was in the old row and is still in the new row, it's fine as long as it's not in the *other* rows.
                # all_used_except_this_row contains diffs from OTHER rows.
                # So we just check: new_diffs & all_used_except_this_row == empty
                if new_diffs & all_used_except_this_row:
                    continue
                
                # Valid move
                # Check if it improves the global max (which is just the max of this row if this row has the global max)
                new_row_max = max(cand_row)
                if new_row_max < best_max:
                    best_row = cand_row
                    best_max = new_row_max
                    improved = True
                    break # Restart loop with new row
                elif new_row_max == best_max:
                    # Accept neutral moves to escape local minima? 
                    # For a deterministic shrink, maybe not. 
                    # But for ILS, we want to find *a* valid smaller max.
                    # Let's just take the first improvement.
                    pass

    return best_row

def _local_search(rows, k, all_used):
    """
    Perform local search to minimize the global maximum.
    Focuses on rows with the current global maximum.
    """
    current_max = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    # Iterate until no improvement
    improved = True
    while improved:
        improved = False
        # Identify rows with the current max
        max_rows = [i for i in range(len(rows)) if row_maxes[i] == current_max]
        
        if not max_rows:
            break
            
        # Shuffle to avoid bias
        random.shuffle(max_rows)
        
        for ri in max_rows:
            # Compute all_used except this row
            # all_used - row_diffs[ri]
            external_diffs = all_used - row_diffs[ri]
            
            new_row = _try_shrink_row(rows[ri], k, external_diffs)
            
            if max(new_row) < row_maxes[ri]:
                # Update state
                all_used -= row_diffs[ri]
                new_diffs = _row_diffs_set(new_row)
                all_used |= new_diffs
                rows[ri] = new_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = max(new_row)
                current_max = max(row_maxes)
                improved = True
                break # Restart loop as max changed
    
    return rows, current_max

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (150s kill)
    
    # 1. Initialization
    # Use a slightly larger greedy init to ensure we have a valid starting point
    # Standard greedy is usually fine for n=7, k=6.
    rows = _greedy_init(n_rows, k)
    
    # Local search to polish the initial solution
    all_used = _compute_all_used(rows)
    rows, current_max = _local_search(rows, k, all_used)
    
    best_rows = [r[:] for r in rows]
    best_val = current_max
    
    # 2. Iterated Local Search (ILS)
    # We will iterate:
    # 1. Local Search (done above)
    # 2. Perturbation: Pick 2 rows, regenerate them randomly (greedy) avoiding the diffs of the other rows.
    # 3. Local Search again.
    # 4. If improved, keep. Else, revert or keep depending on strategy (standard ILS reverts if not improved, but 
    #    sometimes "accepting worse" helps. Let's stick to strict improvement for stability first, 
    #    or use a simple acceptance criterion).
    
    # Let's use a simple acceptance: if new_val < best_val, accept.
    # If new_val == best_val, accept with prob 0.5 (to explore).
    # If new_val > best_val, reject (revert).
    
    while time.time() < deadline:
        # Perturbation
        # Pick 2 distinct rows to regenerate
        if n_rows < 2:
            break
            
        idx1, idx2 = random.sample(range(n_rows), 2)
        
        # Save current state
        old_rows = [r[:] for r in rows]
        old_all_used = set(all_used)
        
        # Remove diffs of idx1 and idx2 from all_used to allow new rows to use them
        d1 = _row_diffs_set(rows[idx1])
        d2 = _row_diffs_set(rows[idx2])
        
        external_diffs = all_used - d1 - d2
        
        # Regenerate row 1
        # We need a greedy that avoids external_diffs.
        # _greedy_init currently builds a set of rows. We need a single row generator.
        
        def _gen_single_row(k, forbidden_diffs, start_val=0):
            row = [start_val]
            x = start_val
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & forbidden_diffs:
                    row.append(x)
                    forbidden_diffs |= new_diffs
                else:
                    # If stuck, try a random jump?
                    # For now, just increment.
                    pass
            return row
        
        try:
            new_row1 = _gen_single_row(k, set(external_diffs))
            # Update forbidden for row 2
            ext_diffs_for_row2 = external_diffs | _row_diffs_set(new_row1)
            new_row2 = _gen_single_row(k, set(ext_diffs_for_row2))
        except Exception:
            # If generation fails (infinite loop or error), revert
            continue
            
        # Check if the new rows are valid (no internal conflicts, no external conflicts)
        # _gen_single_row ensures internal validity.
        # It ensures no conflict with forbidden_diffs (which includes other rows and the first new row).
        # So they should be valid.
        
        # Update rows
        rows[idx1] = new_row1
        rows[idx2] = new_row2
        
        # Recompute all_used
        all_used = _compute_all_used(rows)
        
        # Local Search
        rows, new_val = _local_search(rows, k, all_used)
        
        # Acceptance
        if new_val < best_val:
            best_val = new_val
            best_rows = [r[:] for r in rows]
        elif new_val == best_val:
            if random.random() < 0.5:
                # Accept to explore
                pass # rows are already updated
            else:
                # Revert
                rows = [r[:] for r in old_rows]
                all_used = old_all_used
        else:
            # Revert
            rows = [r[:] for r in old_rows]
            all_used = old_all_used
            
        # If we hit a dead end (no improvement for a long time), we might want to restart from best
        # But ILS usually handles this via the perturbation.
        
    return best_rows

# EVOLVE-BLOCK-END
