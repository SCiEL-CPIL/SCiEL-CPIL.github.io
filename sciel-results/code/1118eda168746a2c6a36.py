import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    """
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Precompute total number of differences needed
    # Each row has (k+1)*k/2 differences.
    # Total diffs = n_rows * (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    # The counting bound is roughly sqrt(2 * total_diffs) but we know the best known is 134 for (4,7).
    # Let's start from a reasonable lower bound.
    # For (4,7), total_diffs = 4 * 28 = 112. sqrt(2*112) ~ 15. Wait, the bound is on the max entry M.
    # The differences are in [1, M]. So M must be at least total_diffs.
    # Actually, the differences are distinct, so they must be 1..total_diffs.
    # Thus max difference is at least total_diffs.
    # Since max difference is M (if 0 is min and M is max), M >= total_diffs.
    # For (4,7), M >= 112.
    
    min_scope = total_diffs
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def solve_csp(n_rows, k, max_scope, rng, time_limit=5.0):
        """
        Attempt to construct rows with max entry <= max_scope using CSP with forward checking.
        """
        start_time = time.time()
        
        # State: rows is a list of lists. We build them one by one? Or interleaved?
        # Interleaved might be better for global constraint checking.
        # Let's build row by row, but use random restarts.
        
        # Actually, a better CSP approach:
        # Variables: x[r][j] for r in 0..n_rows-1, j in 0..k.
        # x[r][0] = 0.
        # Constraints: x[r][j] < x[r][j+1], and all diffs unique.
        
        # Heuristic: Order variables by "most constrained" first.
        # However, implementing full CSP with backtracking in Python within time limits is tricky.
        # Let's try a randomized greedy with "lookahead" (forward checking).
        
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        # We will fill positions j=1..k for all rows.
        # To allow for global optimization, we might want to pick which row to extend next.
        
        # List of (row_idx, current_pos)
        # Initially all rows are at pos 1 (since pos 0 is 0).
        
        # To ensure we don't get stuck, we use a randomized order of extending rows.
        
        # However, standard greedy often fails because early choices block later ones.
        # Randomized restarts are key.
        
        # Let's try a different approach: 
        # Generate random candidate sets for each row from [0, max_scope] and check validity?
        # Too many candidates.
        
        # Let's stick to a smarter greedy:
        # At each step, pick the row with the fewest valid next steps (MRV heuristic).
        
        active_rows = list(range(n_rows))
        
        while any(len(rows[r]) < k + 1 for r in range(n_rows)):
            if time.time() - start_time > time_limit:
                return None
            
            # Calculate valid next steps for each active row
            candidates = {}
            for r in active_rows:
                if len(rows[r]) >= k + 1:
                    continue
                last_val = rows[r][-1]
                valid_next = []
                # Try values from last_val + 1 to max_scope
                # We can optimize by checking diffs
                for v in range(last_val + 1, max_scope + 1):
                    new_diffs = {v - a for a in rows[r]}
                    if not (new_diffs & used_diffs):
                        valid_next.append(v)
                
                if valid_next:
                    candidates[r] = valid_next
            
            if not candidates:
                return None
            
            # MRV: Pick row with fewest candidates
            # Tie-breaking: Random
            min_count = min(len(v) for v in candidates.values())
            min_rows = [r for r, v in candidates.items() if len(v) == min_count]
            r = rng.choice(min_rows)
            
            # Choose a value from candidates[r]
            # Heuristic: Pick the smallest value to keep scope low? 
            # Or random? 
            # Let's try a mix: biased towards smaller values but with randomization.
            vals = candidates[r]
            # Weighted random choice, favoring smaller values
            # Simple: just pick randomly from the first few?
            # Or just pick the smallest to minimize scope immediately?
            # Picking smallest is greedy. Picking random is stochastic.
            # Let's pick randomly from the valid set, but maybe bias towards lower half.
            if rng.random() < 0.5:
                # Pick from lower half of valid values
                half = len(vals) // 2
                if half > 0:
                    v = rng.choice(vals[:half])
                else:
                    v = vals[0]
            else:
                v = rng.choice(vals)
            
            rows[r].append(v)
            new_diffs = {v - a for a in rows[r][:-1]}
            used_diffs |= new_diffs
            
            # Remove r from active if complete
            if len(rows[r]) == k + 1:
                active_rows.remove(r)
                
        return rows

    def local_search(rows, n_rows, k, rng, time_limit=5.0):
        """
        Local search to minimize max entry.
        """
        start_time = time.time()
        rows = [row[:] for row in rows]
        current_max = get_max_entry(rows)
        
        # Precompute diffs
        def get_all_diffs(rows):
            diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        diffs.add(row[j]-row[i])
            return diffs
        
        all_diffs = get_all_diffs(rows)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to reduce each element
            for r in range(n_rows):
                for pos in range(k, 0, -1):
                    old_val = rows[r][pos]
                    # Try values from rows[r][pos-1]+1 to old_val-1
                    for new_val in range(rows[r][pos-1]+1, old_val):
                        # Check validity
                        # Remove old diffs involving pos
                        old_row = rows[r]
                        old_diffs_involved = set()
                        for i in range(pos):
                            old_diffs_involved.add(old_row[pos] - old_row[i])
                        for i in range(pos+1, len(old_row)):
                            old_diffs_involved.add(old_row[i] - old_row[pos])
                        
                        # New diffs
                        new_row = old_row[:]
                        new_row[pos] = new_val
                        new_diffs_involved = set()
                        for i in range(pos):
                            new_diffs_involved.add(new_row[pos] - new_row[i])
                        for i in range(pos+1, len(new_row)):
                            new_diffs_involved.add(new_row[i] - new_row[pos])
                        
                        # Check conflicts
                        # Same row other diffs
                        same_row_other = set()
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                if i != pos and j != pos:
                                    same_row_other.add(new_row[j] - new_row[i])
                        
                        if new_diffs_involved & same_row_other:
                            continue
                        if new_diffs_involved & (all_diffs - old_diffs_involved):
                            continue
                        
                        # Valid
                        rows[r][pos] = new_val
                        all_diffs = get_all_diffs(rows)
                        new_max = get_max_entry(rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        break # Move to next pos/row
                    
                    if improved:
                        break
                if improved:
                    break
        return rows, current_max

    best_rows = None
    best_score = float('inf')
    
    # Start from counting bound
    current_scope = min_scope
    
    # Strategy: Iterative deepening on scope
    # Try to find a solution with scope = min_scope, then min_scope+1, etc.
    # Use randomized restarts for each scope.
    
    max_scope_to_try = min_scope + 50 # Don't go too high initially
    
    while current_scope <= max_scope_to_try and time.time() < deadline:
        # Try multiple times for this scope
        attempts = 10
        found = False
        for i in range(attempts):
            if time.time() > deadline:
                break
            rng = random.Random(i * 1000 + current_scope)
            rows = solve_csp(n_rows, k, current_scope, rng, time_limit=2.0)
            if rows is not None:
                if is_valid(rows):
                    s = get_max_entry(rows)
                    if s < best_score:
                        best_score = s
                        best_rows = [r[:] for r in rows]
                    found = True
                    # If we found a solution, we can try to improve it with local search
                    if time.time() < deadline:
                        ls_rows, ls_score = local_search(best_rows, n_rows, k, random.Random(123), time_limit=2.0)
                        if ls_score < best_score:
                            best_score = ls_score
                            best_rows = [r[:] for r in ls_rows]
        if found:
            break # Found solution for this scope, stop deepening
        
        current_scope += 1
    
    # If no solution found in the tight range, do a general search
    if best_rows is None:
        # Fallback to broader search
        for scope in range(min_scope + 50, min_scope + 150):
            if time.time() > deadline:
                break
            rng = random.Random(scope)
            rows = solve_csp(n_rows, k, scope, rng, time_limit=2.0)
            if rows is not None and is_valid(rows):
                s = get_max_entry(rows)
                if s < best_score:
                    best_score = s
                    best_rows = [r[:] for r in rows]
                break
    
    # Final polish with local search if time remains
    if best_rows is not None and time.time() < deadline:
        # Run local search for remaining time
        time_left = deadline - time.time() - 2
        if time_left > 1:
            ls_rows, ls_score = local_search(best_rows, n_rows, k, random.Random(999), time_limit=time_left)
            if ls_score < best_score:
                best_score = ls_score
                best_rows = [r[:] for r in ls_rows]
    
    if best_rows is not None:
        return best_rows
    
    # Absolute fallback: simple greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
