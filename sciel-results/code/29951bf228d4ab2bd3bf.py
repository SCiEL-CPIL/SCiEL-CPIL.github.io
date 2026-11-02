# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    
    Approach: Local Search / Hill Climbing
    1. Generate an initial valid solution using deterministic greedy.
    2. Iteratively try to reduce individual elements to lower the maximum.
    3. Use random restarts/perturbations if stuck.
    """
    start_time = time.time()
    time_limit = 110.0
    
    # 1. Initial valid solution
    # We use a deterministic greedy to get a valid baseline quickly.
    # This is much faster than random greedy for getting *a* solution.
    best_rows = _greedy_deterministic(n_rows, k)
    if not best_rows:
        return [] # Should not happen for reasonable inputs
        
    best_max = max(max(r) for r in best_rows)
    
    # Precompute used diffs for the best solution
    # We will maintain a global set of used diffs for the current best solution
    # and update it incrementally.
    
    # Helper to compute used diffs
    def get_used_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs

    current_rows = [r[:] for r in best_rows]
    current_used_diffs = get_used_diffs(current_rows)
    
    # Local Search Loop
    rng = random.Random(42)
    no_improvement_count = 0
    max_no_improvement = 5000 # Restart after this many failed attempts
    
    while time.time() - start_time < time_limit:
        if no_improvement_count > max_no_improvement:
            # Perturbation: Rebuild a random row greedily
            # Remove its diffs from used set
            row_idx = rng.randint(0, n_rows - 1)
            old_row = current_rows[row_idx]
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    current_used_diffs.remove(old_row[j] - old_row[i])
            
            # Rebuild row
            new_row = _greedy_single_row(current_rows, row_idx, current_used_diffs, k, rng)
            if new_row:
                current_rows[row_idx] = new_row
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        current_used_diffs.add(new_row[j] - new_row[i])
                # Check if this perturbation improved or kept the max low
                new_max = max(max(r) for r in current_rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                no_improvement_count = 0
            else:
                # If rebuild failed, restore old row (shouldn't happen if greedy is robust, but safe)
                current_rows[row_idx] = old_row
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        current_used_diffs.add(old_row[j] - old_row[i])
                no_improvement_count = 0
            continue

        # Try to improve by reducing a random element
        # Pick a random row and a random position > 0
        r_idx = rng.randint(0, n_rows - 1)
        c_idx = rng.randint(1, k) # Positions 1..k (0 is fixed at 0)
        
        row = current_rows[r_idx]
        old_val = row[c_idx]
        
        # Determine valid range for new value
        # Must be > row[c_idx-1] and < row[c_idx+1] (if exists)
        min_val = row[c_idx-1] + 1
        max_val = row[c_idx+1] - 1 if c_idx < k else old_val - 1
        
        if min_val > max_val:
            no_improvement_count += 1
            continue
            
        # We want to try values smaller than old_val
        # To be efficient, we can sample a few candidates or scan downwards
        # Scanning downwards from old_val - 1 to min_val
        # But we should check if it reduces the global max.
        # If old_val < best_max, reducing it doesn't help the global max directly,
        # but it might free up space? No, it just uses smaller diffs.
        # The primary goal is to reduce the global max.
        # So we should prioritize reducing elements that are == best_max.
        
        # Strategy:
        # 1. If there are elements equal to best_max, try to reduce one of them.
        # 2. Otherwise, try to reduce a random element (might allow a better configuration later? 
        #    Actually, reducing any element makes the set "tighter", which is generally good).
        
        # Let's identify candidates for reduction
        # We will try to reduce `old_val` to some `new_val < old_val`.
        
        # To speed up, we don't need to check every value.
        # We can just pick a random `new_val` in [min_val, old_val - 1].
        # Or scan from min_val upwards to find the *smallest* valid replacement.
        # Finding the smallest valid replacement is best for local search.
        
        found_improvement = False
        # We only care about reducing the value.
        # If old_val is the current max, reducing it is good.
        # If old_val is not the max, reducing it is still good (makes the row tighter).
        
        # Let's try to find the smallest valid x in [min_val, old_val - 1]
        # If we find one, we replace old_val with x.
        
        # Optimization: If old_val > best_max, we MUST reduce it to < best_max to improve.
        # If old_val <= best_max, reducing it might help future steps.
        
        # We will scan from min_val to old_val - 1.
        # This is O(k^2) per check, and we do it for each candidate.
        # k is small (e.g., 5-10). n_rows is small.
        
        # Pre-calculate diffs involving row[r_idx]
        # Diffs to remove: {old_val - row[i] for i < c_idx} U {row[j] - old_val for j > c_idx}
        diffs_to_remove = set()
        for i in range(c_idx):
            diffs_to_remove.add(old_val - row[i])
        for j in range(c_idx + 1, len(row)):
            diffs_to_remove.add(row[j] - old_val)
        
        # Temporarily remove them from global set? 
        # Better: Check against (current_used_diffs - diffs_to_remove)
        
        valid_replacement = None
        for x in range(min_val, old_val):
            # Check if x is valid
            # New diffs: {x - row[i] for i < c_idx} U {row[j] - x for j > c_idx}
            # These must not be in (current_used_diffs - diffs_to_remove)
            
            # Also, new diffs must be distinct within the row (guaranteed by strict increase)
            
            # Quick check: if x - row[i] is in diffs_to_remove, it's fine (it was there before? No, diffs_to_remove are OLD diffs).
            # Wait, if x - row[i] == old_val - row[i], that implies x == old_val, which is not the case.
            # So new diffs are disjoint from old diffs by value? Not necessarily.
            # Example: row = [0, 2, 5]. old_val=5. diff 5-2=3.
            # New x=3. diff 3-2=1. diff 0-3? No.
            # The set of diffs changes.
            
            # We need to ensure new_diffs are not in (current_used_diffs - diffs_to_remove).
            
            valid = True
            for i in range(c_idx):
                d = x - row[i]
                # d is new. It must not be in current_used_diffs UNLESS it was in diffs_to_remove?
                # No, if d was in diffs_to_remove, it means some other pair in the SAME row had that diff?
                # No, diffs_to_remove are the diffs involving old_val.
                # If d (new diff) is in current_used_diffs, is it allowed?
                # Only if it was one of the diffs_to_remove?
                # If d == old_diff, and old_diff is being removed, then it's okay to reuse it?
                # No, the difference must be unique across ALL rows.
                # If d was used by old_val, and we remove old_val's diffs, d becomes available.
                # So if d is in diffs_to_remove, it is available.
                # If d is in current_used_diffs but NOT in diffs_to_remove, it is taken by another row or another pair in this row.
                
                if d in current_used_diffs and d not in diffs_to_remove:
                    valid = False
                    break
            if valid:
                for j in range(c_idx + 1, len(row)):
                    d = row[j] - x
                    if d in current_used_diffs and d not in diffs_to_remove:
                        valid = False
                        break
            
            if valid:
                valid_replacement = x
                break # We want the smallest valid x to keep the row tight
        
        if valid_replacement is not None:
            # Apply the change
            row[c_idx] = valid_replacement
            
            # Update used_diffs
            current_used_diffs -= diffs_to_remove
            for i in range(c_idx):
                current_used_diffs.add(valid_replacement - row[i])
            for j in range(c_idx + 1, len(row)):
                current_used_diffs.add(row[j] - valid_replacement)
            
            new_max = max(max(r) for r in current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in current_rows]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    return best_rows

def _greedy_deterministic(n_rows, k):
    """Original greedy construction."""
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

def _greedy_single_row(all_rows, exclude_idx, used_diffs, k, rng):
    """
    Construct a single row for index exclude_idx, avoiding diffs in used_diffs.
    This is used for perturbation.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used_diffs:
            row.append(x)
            # We don't add to used_diffs here because the caller manages the global set
            # But wait, the row itself must have distinct diffs.
            # Greedy ensures that.
            break
    if len(row) != k + 1:
        return None
    return row

# EVOLVE-BLOCK-END
