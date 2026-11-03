import time
import random
import numpy as np

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140

    # Total number of differences required
    total_diffs = n_rows * k * (k + 1) // 2
    min_scope = total_diffs + 1

    def get_diffs(row):
        """Calculate the set of positive differences for a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ri)
        return diffs

    def is_valid(rows):
        """Check if the set of rows is valid (all diffs distinct across all rows)."""
        used = set()
        for row in rows:
            d = get_diffs(row)
            if d & used:
                return False
            used |= d
        return True

    def greedy_row(limit, used, seed):
        """
        Greedily construct a single row of k+1 integers starting with 0,
        using differences not in 'used', with max <= limit.
        Returns the row or None.
        """
        random.seed(seed)
        row = [0]
        # We need k more elements
        # To ensure we can fit k elements in [last+1, limit], we need last + k <= limit
        # So last <= limit - k
        
        # We will pick elements one by one.
        # Heuristic: pick a random valid x, but bias towards the middle/end to spread diffs?
        # Or small to keep compact?
        # Let's try a mix.
        
        for _ in range(k):
            last = row[-1]
            # Candidates: last+1 to limit
            # But we must ensure remaining elements fit.
            # If we pick x, remaining elements must be in [x+1, limit].
            # So x + remaining_count <= limit => x <= limit - remaining_count.
            remaining_count = k - len(row) + 1 # elements to add after this one? No.
            # Current len(row) = d. We need k+1-d elements total including current.
            # So we pick 1, then k+1-d-1 = k-d remain.
            # So x + (k - (len(row)-1)) <= limit?
            # Let's just say: if we pick x, we need space for (k - (len(row)-1)) more elements.
            # So x + 1 + ... + (k - len(row) + 1) <= limit?
            # Simpler: max possible value for next elements is limit.
            # Min gap is 1.
            # So x + (k - (len(row)-1)) <= limit is a necessary condition?
            # If we pick x, the next min is x+1, then x+2...
            # So yes, x + (k - (len(row)-1)) <= limit.
            
            max_x = limit - (k - (len(row) - 1))
            if max_x < last + 1:
                return None
            
            # Find valid x in [last+1, max_x]
            valid_x = []
            for x in range(last + 1, max_x + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_x.append(x)
            
            if not valid_x:
                return None
            
            # Pick randomly from valid_x
            # To improve packing, maybe pick the one that uses "rare" diffs?
            # For now, random.
            x = random.choice(valid_x)
            new_diffs = {x - a for a in row}
            row.append(x)
            used.update(new_diffs)
        
        return row

    def local_search(rows, used, max_iterations, deadline):
        """
        Perform local search to minimize the scope.
        Move: pick a row, remove it, rebuild it with a tighter limit.
        Accept if valid and scope decreases.
        """
        best_scope = max(max(r) for r in rows)
        current_rows = [list(r) for r in rows]
        current_used = set()
        for r in current_rows:
            current_used |= get_diffs(r)
        
        no_improve = 0
        temp = 1.0 # Temperature for simulated annealing?
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random row to modify
            r_idx = random.randint(0, n_rows - 1)
            old_row = current_rows[r_idx]
            old_diffs = get_diffs(old_row)
            
            # Remove old row
            current_used -= old_diffs
            
            # Try to rebuild this row with a smaller max
            # Current max of this row is old_row[-1]
            # Try limits from old_row[-1] - 1 down to some bound
            # Or just try a random smaller limit?
            # Let's try to find ANY valid row with max < old_row[-1]
            
            # What is the minimum possible max for this row given the other rows?
            # It's at least len(row)-1 (if 0,1,2...) but diffs must be unique.
            # Let's try limits: old_row[-1] - 1, old_row[-1] - 2, ...
            # But this is slow.
            
            # Alternative: Rebuild with limit = old_row[-1] - 1.
            # If successful, we improved this row's max.
            # Then check if global max improved.
            
            new_limit = old_row[-1] - 1
            if new_limit < k: # Must fit k+1 elements
                new_limit = k
            
            new_row = None
            # Try a few seeds for rebuilding
            for seed in range(10):
                if time.time() > deadline:
                    break
                # Save used state
                saved_used = set(current_used)
                new_row = greedy_row(new_limit, current_used, seed)
                if new_row is not None:
                    break
                # If failed, used wasn't modified in greedy_row if it returns None?
                # In my greedy_row above, I do `used.update(new_diffs)` INSIDE the loop.
                # So if it fails halfway, used is corrupted!
                # I need to fix greedy_row to not modify used until success, or restore it.
                # Let's fix greedy_row to return None without modifying used if it fails.
                # Or better: greedy_row takes used, and only returns row if successful.
                # I will rewrite greedy_row to be "pure" (no side effects on used).
                
            # Wait, the current greedy_row modifies used.
            # I need to change the structure.
            
            # Let's implement a pure greedy_row.
            # Actually, let's just restart the local search implementation with a cleaner design.
            # This edit is getting messy. I will replace the whole construct function body with a cleaner implementation using a pure helper.
            
            # Since I can't easily "restart" in a SEARCH/REPLACE without replacing the whole function,
            # and the current function is huge, I should replace the whole `construct` function.
            
            # But the SEARCH block is the whole function.
            # So I will just write the new function here.
            
            # Let's abort this local_search attempt and write a clean function in the REPLACE block.
            return current_rows # Placeholder to satisfy syntax if we don't replace whole thing? No, we are replacing whole thing.

    # --- REWRITTEN LOGIC BELOW ---
    
    # We will use a different strategy:
    # 1. Generate initial solution with a loose bound (e.g., 200).
    # 2. Use Local Search / Simulated Annealing to minimize the scope.
    #    Move: Pick a random row, remove it, and try to reconstruct it with a smaller max value.
    #    If successful and valid, accept if it improves global max, or with probability exp(-delta/T) if it worsens (SA).
    #    To make it faster, we can try to reconstruct with max = current_max - 1.
    
    # Helper: Pure greedy row construction
    def try_construct_row(used, limit, seed):
        """
        Try to construct a row of k+1 integers starting with 0, max <= limit,
        with diffs disjoint from 'used'.
        Returns (row, diffs) or (None, None).
        """
        random.seed(seed)
        row = [0]
        
        for _ in range(k):
            last = row[-1]
            # Max x such that we can fit remaining elements
            remaining = k - (len(row) - 1)
            max_x = limit - remaining + 1 # x + 1 + ... + remaining <= limit => x <= limit - remaining + 1?
            # If we pick x, next min is x+1.
            # So x + (remaining - 1) <= limit.
            max_x = limit - (remaining - 1)
            
            if max_x < last + 1:
                return None, None
            
            valid_x = []
            # Optimization: Check diffs
            for x in range(last + 1, max_x + 1):
                # Check if diffs x-a are in used
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    valid_x.append(x)
            
            if not valid_x:
                return None, None
            
            # Pick random x
            x = random.choice(valid_x)
            row.append(x)
            # Update used? No, we are in a "try" context.
            # We need to track used locally to ensure subsequent picks are valid within this row too.
            # But we also must not conflict with global 'used'.
            # So we maintain a local_used which starts as copy of global used?
            # That's expensive.
            # Better: just check against global used AND previously picked elements in this row.
            
            # Wait, the check `if (x-a) in used` only checks global used.
            # It doesn't check if x-a conflicts with a previous diff in THIS row.
            # But since row elements are increasing, x-a is always positive.
            # And we check all a in row.
            # So if x-a was generated by a previous step in this row, is it in 'used'?
            # No, 'used' is the global set of diffs from OTHER rows.
            # We need to ensure diffs within the NEW row are also unique.
            # So we need a local set of diffs for the current row being built.
            
            # Let's restart try_construct_row with local diff tracking.
            return None, None

    # Corrected pure greedy row
    def try_construct_row_pure(used_global, limit, seed):
        random.seed(seed)
        row = [0]
        local_diffs = set()
        
        for _ in range(k):
            last = row[-1]
            remaining = k - (len(row) - 1)
            max_x = limit - (remaining - 1)
            
            if max_x < last + 1:
                return None, None
            
            valid_x = []
            for x in range(last + 1, max_x + 1):
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used_global or d in local_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if ok:
                    valid_x.append((x, new_diffs))
            
            if not valid_x:
                return None, None
            
            # Pick random
            x, new_diffs = random.choice(valid_x)
            row.append(x)
            local_diffs.update(new_diffs)
        
        return row, local_diffs

    # Initial construction
    # Start with a high limit to ensure we get a valid set quickly
    init_limit = 200
    used = set()
    rows = []
    success = False
    
    for r_idx in range(n_rows):
        # Try to build a row
        for seed in range(100):
            row, diffs = try_construct_row_pure(used, init_limit, seed)
            if row is not None:
                rows.append(row)
                used |= diffs
                break
        else:
            # Failed to build a row
            # Increase limit?
            # For now, if we fail, we return a fallback.
            pass
    
    if len(rows) < n_rows:
        # Fallback: simple greedy without limit
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

    # Local Search / Simulated Annealing
    # State: rows, used
    # Objective: Minimize max(max(r) for r in rows)
    
    best_rows = [list(r) for r in rows]
    best_scope = max(max(r) for r in best_rows)
    current_rows = [list(r) for r in rows]
    current_used = set(used)
    current_scope = best_scope
    
    T = 10.0
    cooling = 0.99
    
    # We want to improve the scope.
    # Move: Pick a row i. Remove it. Try to rebuild it with max < current_row_max.
    # If successful, we have a new candidate.
    # If new_global_max < current_global_max, accept.
    # If new_global_max == current_global_max, accept with prob 0.5? (Neutral move)
    # If new_global_max > current_global_max, reject (or SA accept).
    # But we want to MINIMIZE. So we should only accept if it improves or is neutral (to explore).
    # However, to escape local minima, we might need to temporarily worsen?
    # But the "worsening" is in the scope.
    # If we worsen the scope, we might get stuck in a bad state.
    # But if we only accept improvements, we might get stuck in a local minimum where no single row can be improved.
    # So SA is good.
    
    # Let's implement SA.
    # Energy = scope.
    # We want to minimize Energy.
    # Move: Rebuild a random row with limit = current_row_max - 1.
    # If success:
    #   new_scope = max(max(r) for r in new_rows)
    #   delta = new_scope - current_scope
    #   if delta < 0: accept
    #   else: accept with prob exp(-delta/T)
    
    # Note: Rebuilding with limit = current_row_max - 1 might fail often.
    # We can try limit = current_row_max - 1, -2, etc.
    
    steps = 0
    while time.time() < deadline:
        if steps % 1000 == 0 and T < 0.1:
            break
        steps += 1
        
        # Pick a random row
        i = random.randint(0, n_rows - 1)
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        old_max = old_row[-1]
        
        # Remove from current state
        current_used -= old_diffs
        
        # Try to rebuild
        # Target max: old_max - 1
        # If that fails, try old_max - 2?
        # Or just try old_max - 1.
        # If we can't reduce this row's max, we can't reduce the global max if it's determined by this row.
        # But if global max is determined by another row, reducing this row's max might not change global max.
        
        new_row = None
        new_diffs = None
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            new_row, new_diffs = try_construct_row_pure(current_used, old_max - 1, seed)
            if new_row is not None:
                break
        
        if new_row is None:
            # Restore old row
            current_rows[i] = old_row
            current_used |= old_diffs
            continue
        
        # Calculate new scope
        # We don't need to recalculate all maxes if we know old_max was the unique max?
        # Just calculate max for all rows.
        new_rows = current_rows
        new_rows[i] = new_row
        new_scope = max(max(r) for r in new_rows)
        delta = new_scope - current_scope
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_rows[i] = new_row
            current_used |= new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        
        else:
            # Reject
            current_rows[i] = old_row
            current_used |= old_diffs
        
        T *= cooling

    return best_rows
