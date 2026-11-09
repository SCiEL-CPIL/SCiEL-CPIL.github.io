import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    best_rows = None
    best_max = float('inf')
    
    # 1. Initial Greedy Solution
    # This provides a baseline upper bound.
    # We try a few variations of greedy (though standard greedy is deterministic, 
    # we can simulate diversity by slightly randomizing the choice among valid 
    # small candidates in a "relaxed greedy" mode, but standard greedy is usually 
    # a strong starting point for the backtracker to improve upon).
    
    def get_greedy_solution():
        used_diffs = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check conflicts with used_diffs
                # Diffs are x - a for a in row
                conflict = False
                for a in row:
                    d = x - a
                    if (used_diffs >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    # Update used_diffs
                    for a in row[:-1]: # a is in row, x is new last
                        d = x - a
                        used_diffs |= (1 << d)
            rows.append(row)
        return rows

    greedy_rows = get_greedy_solution()
    best_max = max(max(r) for r in greedy_rows)
    best_rows = [r[:] for r in greedy_rows]
    
    # 2. Backtracking Search to improve best_max
    # We try to find a solution with max < best_max.
    # We use bitmasks for efficiency.
    
    def try_construct(target_max, n_rows, k, rng):
        used_diff_mask = 0
        rows = []
        
        def backtrack_row(row, used_diff_mask):
            # row is the current row being built
            # used_diff_mask is the global mask of used diffs
            if len(row) == k + 1:
                return row[:]
            
            last = row[-1]
            # We need to pick the next element x > last
            # x can be at most target_max
            # Also, we need enough room to fill the rest of the row?
            # If we pick x, the remaining elements must be > x and <= target_max.
            # This is a weak constraint, but let's just iterate.
            
            valid_candidates = []
            # Optimization: Iterate from last+1 to target_max
            # We prefer smaller x to keep the row "compact", which helps fit more rows.
            # But we randomize to escape local minima.
            
            for x in range(last + 1, target_max + 1):
                # Check if adding x causes conflicts
                conflict = False
                for a in row:
                    d = x - a
                    if (used_diff_mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            # Heuristic: Try a subset of the smallest candidates, shuffled.
            # Taking too many candidates increases branching factor.
            # Taking too few might miss solutions.
            # Let's take up to 10 smallest valid candidates.
            W = 10
            candidates_to_try = valid_candidates[:W]
            rng.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                new_row = row + [x]
                # Calculate new diffs mask
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                
                res = backtrack_row(new_row, used_diff_mask | new_diffs_mask)
                if res is not None:
                    return res
                    
            return None

        # Construct rows one by one
        for i in range(n_rows):
            row = [0]
            res = backtrack_row(row, used_diff_mask)
            if res is None:
                return None
            # Update global mask with all diffs in this row
            row_mask = 0
            for a_idx in range(len(res)):
                for b_idx in range(a_idx + 1, len(res)):
                    d = res[b_idx] - res[a_idx]
                    row_mask |= (1 << d)
            used_diff_mask |= row_mask
            rows.append(res)
            
        return rows

    # 3. Local Improvement: Rebuild a specific row to reduce max
    def try_local_improve(rows, target_max, rng):
        """
        Try to rebuild one row to have max <= target_max, 
        given the diffs used by the other rows.
        Returns new rows if successful, else None.
        """
        n = len(rows)
        # Calculate diffs used by all rows except the one we are rebuilding?
        # Actually, we can just pick a row, remove its diffs from the global set, 
        # and try to rebuild it.
        
        # Pick a random row to rebuild
        row_idx = rng.randint(0, n - 1)
        
        # Compute mask of diffs used by OTHER rows
        other_mask = 0
        for i, r in enumerate(rows):
            if i == row_idx:
                continue
            for a_idx in range(len(r)):
                for b_idx in range(a_idx + 1, len(r)):
                    d = r[b_idx] - r[a_idx]
                    other_mask |= (1 << d)
        
        # Try to build row_idx with max <= target_max
        # We reuse the backtrack logic but only for one row
        def backtrack_single_row(row, mask):
            if len(row) == k + 1:
                return row[:]
            last = row[-1]
            valid_candidates = []
            for x in range(last + 1, target_max + 1):
                conflict = False
                for a in row:
                    d = x - a
                    if (mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            W = 10
            candidates_to_try = valid_candidates[:W]
            rng.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                new_row = row + [x]
                new_mask = mask
                for a in row:
                    d = x - a
                    new_mask |= (1 << d)
                
                res = backtrack_single_row(new_row, new_mask)
                if res is not None:
                    return res
            return None

        new_row = backtrack_single_row([0], other_mask)
        if new_row is not None:
            # Check if this actually reduces the max or at least fits
            # The goal is to find a solution with a lower global max.
            # If we rebuild row_idx to have max <= target_max, and other rows are unchanged,
            # the new global max is max(max(other_rows), max(new_row)).
            # We only call this if we hope to reduce the global max.
            new_rows = rows[:]
            new_rows[row_idx] = new_row
            return new_rows
        return None

    # Main Search Loop
    # We try to lower the bound.
    current_target = best_max - 1
    
    while time.time() < deadline and current_target >= k:
        found_improvement = False
        # Multiple restarts for this target
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random()
            # 1. Try full construction
            res = try_construct(current_target, n_rows, k, rng)
            if res is not None:
                current_max = max(max(r) for r in res)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in res]
                    found_improvement = True
                    # If we found a solution, we can immediately try to lower the target further
                    # But let's finish the restart loop or break?
                    # Breaking allows us to update current_target immediately.
                    break
            
            # 2. Try local improvement on the current best_rows
            # If we have a solution with best_max, can we rebuild one of its rows 
            # to fit within current_target (which is < best_max)?
            # This is only useful if current_target < best_max.
            # We are in the loop where current_target = best_max - 1.
            # So if we fail full construction, we can try to see if a local change 
            # to the existing best solution allows a row to be "tighter"?
            # No, local improvement reduces the max of a specific row, but if other rows 
            # have max = best_max, the global max remains best_max unless we rebuild 
            # the rows that define the global max.
            # However, if we have a solution with max M, and we try target M-1, 
            # local improvement on the solution with max M might find a way to 
            # replace a row with one that has max M-1, provided the other rows 
            # don't conflict. But if other rows have max M, the global max is still M.
            # So local improvement only helps if we are trying to reduce the max 
            # of a solution where *all* rows are <= M, and we want to get all <= M-1.
            # Since best_rows have max <= best_max, and we are trying target < best_max,
            # local improvement on best_rows with target = current_target is a valid 
            # strategy to find a new best_rows with max <= current_target.
            
            if best_rows and current_target < best_max:
                improved = try_local_improve(best_rows, current_target, rng)
                if improved:
                    new_max = max(max(r) for r in improved)
                    if new_max <= current_target:
                        best_max = new_max
                        best_rows = [r[:] for r in improved]
                        found_improvement = True
                        break

        if found_improvement:
            current_target = best_max - 1
        else:
            # If we couldn't find a solution for current_target, 
            # we might be stuck. 
            # We could try to increase current_target? No, we want to minimize.
            # If we fail to find a solution for best_max - 1, we stop.
            # Unless we try a different strategy?
            # For now, break if no improvement found after restarts.
            break
            
    return best_rows
