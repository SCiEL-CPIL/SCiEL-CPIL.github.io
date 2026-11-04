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

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, randomize=False, seed_val=None):
    """Greedy construction with optional randomization."""
    if seed_val is not None:
        random.seed(seed_val)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                # Try a random step within reasonable bounds
                # The max step should be large enough to skip conflicts but not too large
                # to keep the scope small.
                # Heuristic: max step around k+1 to 2*(k+1)
                max_step = max(1, int((k + 1) * 1.5))
                found = False
                for _attempt in range(30):
                    step = random.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    # Fallback to sequential search from current x
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows

def _local_search(rows, deadline):
    """
    Local search to minimize the scope.
    We try to reduce the value of each element in each row, or swap elements.
    We maintain a global 'used' set of differences.
    When modifying a row, we check its new diffs against 'used' minus the old diffs of that row.
    """
    rows = [list(r) for r in rows]
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate rows in random order to avoid bias
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for r_idx in row_order:
            if time.time() > deadline:
                break
            
            row = rows[r_idx]
            old_diffs = _row_diffs(row)
            safe_used = used - old_diffs
            
            # Try to reduce each element from right to left
            # We can also try to move elements left, but reduction is primary for scope
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                
                # Try values from lower_bound to old_val - 1
                # We can also try values > old_val if we want to swap, but for scope minimization
                # we primarily want to lower the max.
                # However, sometimes lowering an intermediate element requires raising a previous one?
                # No, strictly increasing.
                
                # Optimization: If old_val is the maximum, we definitely want to lower it.
                # If it's not, lowering it might help future rows or allow compression of later rows.
                
                # Search for a valid new_val
                found_new_val = False
                # Try a few random candidates first? Or sequential?
                # Sequential from lower_bound is good for finding the smallest possible value.
                # But maybe a larger value is needed to allow other rows to be smaller?
                # For now, try to minimize this specific element.
                
                # To speed up, if old_val is large, we can jump.
                # But checking each value is O(k) for diff calculation.
                # k is 5, so it's small.
                
                # Try values in a random order mixed with sequential?
                # Let's try sequential from lower_bound up to old_val.
                # If we find one, we take it.
                
                # Actually, we can also try to swap row[pos] with row[pos-1]? No, must be increasing.
                
                # Let's try to reduce row[pos].
                for new_val in range(lower_bound, old_val):
                    # Check if new_val creates conflicts
                    # New diffs are {new_val - row[i] for i in range(pos)}
                    # We must ensure these are not in safe_used.
                    # We must also ensure they don't conflict with other diffs in the SAME row that don't involve pos.
                    # Diffs not involving pos: {row[j] - row[i] for i < j, i,j != pos}
                    
                    # Pre-compute diffs not involving pos?
                    # It's cheap enough to compute on the fly for k=5.
                    
                    conflict = False
                    for i in range(pos):
                        d = new_val - row[i]
                        if d in safe_used:
                            conflict = True
                            break
                        # Check against other diffs in this row
                        # This is the expensive part if done naively.
                        # Let's just check against safe_used first.
                        # If it passes safe_used, we must check against internal row diffs.
                        # Internal row diffs not involving pos are fixed.
                        # Let's pre-compute internal_diffs_not_involving_pos once per pos.
                    
                    if not conflict:
                        # Now check against internal row diffs
                        # We need to ensure {new_val - row[i] for i < pos} is disjoint from
                        # {row[j] - row[i] for i < j, i,j != pos}
                        
                        # Let's build the set of new diffs
                        new_diffs_set = {new_val - row[i] for i in range(pos)}
                        
                        # Build set of old internal diffs not involving pos
                        # This is constant for a given pos and row.
                        # We can pre-compute this outside the loop if we want.
                        # For k=5, it's trivial.
                        
                        internal_conflict = False
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                if i == pos or j == pos:
                                    continue
                                d = row[j] - row[i]
                                if d in new_diffs_set:
                                    internal_conflict = True
                                    break
                            if internal_conflict:
                                break
                        
                        if not internal_conflict:
                            # Accept this new_val
                            # Update row
                            row[pos] = new_val
                            # Update used
                            # Remove old diffs involving pos
                            for i in range(pos):
                                used.discard(old_val - row[i])
                            # Add new diffs involving pos
                            for i in range(pos):
                                used.add(new_val - row[i])
                            
                            improved = True
                            found_new_val = True
                            break
                
                # If we didn't find a smaller value, we could try a larger value?
                # That would increase scope, so we don't want that unless it allows
                # subsequent rows to be much smaller.
                # Given the time limit, let's stick to reduction.
                
                # If we reduced, the lower_bound for the next position (pos-1) is unchanged,
                # but the upper bound for pos-1 is now row[pos]-1.
                # The loop structure handles this naturally as we move to pos-1.

    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    # Initial greedy
    rows = _greedy_construct(n_rows, k, randomize=False)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Local search on the initial solution
    if time.time() < deadline - 5.0:
        improved_rows = _local_search(best_rows, deadline - 5.0)
        improved_scope = _scope(improved_rows)
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
    
    # Try multiple randomized restarts
    # Each restart: random greedy -> local search
    # Keep the best one
    
    num_restarts = 0
    while time.time() < deadline - 10.0:
        # Randomized greedy
        # Use a random seed for reproducibility within the run, but different each time
        random.seed() 
        r_rows = _greedy_construct(n_rows, k, randomize=True)
        r_scope = _scope(r_rows)
        
        # If the greedy is already worse than best, skip local search?
        # No, local search might improve it significantly.
        # But if it's much worse, it might take longer.
        # Let's just run local search with a shorter time budget if greedy is bad.
        
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        # Allocate time for local search
        # If r_scope is close to best, give it more time.
        # If r_scope is much worse, give it less time.
        if r_scope > best_scope * 1.2:
            ls_deadline = time.time() + min(2.0, time_left * 0.2)
        else:
            ls_deadline = time.time() + min(5.0, time_left * 0.5)
            
        if ls_deadline > deadline:
            ls_deadline = deadline - 1.0
            
        if ls_deadline > time.time() + 0.5:
            improved_r_rows = _local_search(r_rows, ls_deadline)
            improved_r_scope = _scope(improved_r_rows)
            
            if improved_r_scope < best_scope:
                best_rows = improved_r_rows
                best_scope = improved_r_scope
                # If we found a new best, we might want to run local search on it again?
                # The local search function already iterates until no improvement or deadline.
                # So it should be converged.
                
        num_restarts += 1
        
        # Optional: If we have time, try to compress the current best further
        # by running local search again on best_rows
        if num_restarts % 5 == 0 and time.time() < deadline - 5.0:
            final_improved = _local_search(best_rows, deadline - 2.0)
            final_scope = _scope(final_improved)
            if final_scope < best_scope:
                best_rows = final_improved
                best_scope = final_scope

    return best_rows

# EVOLVE-BLOCK-END
