import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 118.0  # Safety margin
    
    # Theoretical lower bound for scope M
    # Number of differences per row = k * (k - 1) / 2
    diffs_per_row = k * (k - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    lower_bound = total_diffs_needed
    
    # Known best is 112. We search from lower_bound to 112 (or slightly more if needed)
    # If we find a solution for M, it's optimal if M == lower_bound.
    # Otherwise, we return the first M for which we find a solution.
    
    best_rows = None
    best_scope = float('inf')
    
    # Fallback: Simple greedy to ensure we always return a valid solution
    def deterministic_greedy():
        used = 0  # Bitset for used differences
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check conflicts
                conflict = False
                for a in row:
                    diff = x - a
                    if (used >> diff) & 1:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    # Update used bitset
                    for a in row[:-1]:
                        used |= (1 << (x - a))
            rows.append(row)
        return rows

    # Initialize with greedy solution to have a fallback
    fallback_rows = deterministic_greedy()
    fallback_scope = max(max(r) for r in fallback_rows)
    
    # If greedy is already very good (e.g., <= 112), we might still try to improve
    # But if it's 112, we can try to find 105-111.
    
    # Main Solver: Randomized Backtracking with Bitset
    def solve_scope(M, time_budget):
        """
        Attempts to find a solution with max element <= M.
        Returns list of rows or None.
        """
        used = 0
        rows = []
        
        # Symmetry breaking: rows are ordered by their max element.
        # prev_max is the max element of the last completed row.
        # For the first row, prev_max = -1.
        
        # Heuristic: To improve search, we can sort candidates.
        # However, dynamic sorting is expensive. We rely on randomization and restarts.
        
        def backtrack(row_idx, current_row, last_val, prev_max):
            nonlocal used, rows
            
            # Time check
            if time.time() - start_time > start_time + time_budget:
                return False, "timeout"
                
            if row_idx == n_rows:
                return True, None
                
            if len(current_row) == k + 1:
                # Row complete
                row_max = current_row[-1]
                if row_max < prev_max:
                    return False, None
                
                # Calculate new differences for this row
                new_diffs_mask = 0
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        new_diffs_mask |= (1 << d)
                
                # Check if any conflict with existing used diffs
                if (used & new_diffs_mask) != 0:
                    return False, None
                
                # Commit
                rows.append(current_row)
                used |= new_diffs_mask
                
                ok, reason = backtrack(row_idx + 1, [0], 0, row_max)
                
                if ok:
                    return True, None
                    
                # Backtrack
                rows.pop()
                used ^= new_diffs_mask
                return False, None
            
            # Choose next value x
            start_x = last_val + 1
            
            # If this is the last element to be added, it must be >= prev_max
            # Actually, the row max is the last element. So if len(current_row) == k,
            # the next x is the max, so x >= prev_max.
            # If len(current_row) < k, x can be anything > last_val, 
            # but we must ensure it's possible to reach >= prev_max later.
            
            candidates = []
            for x in range(start_x, M + 1):
                # Quick check: does x create immediate conflicts?
                # Diffs are x - a for a in current_row
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
            
            if not candidates:
                return False, None
            
            # Heuristic: 
            # We want to choose x that leaves the most options for the rest of the row and future rows.
            # A common heuristic is "least constraining value". 
            # Approximation: prefer x that are smaller? Or larger?
            # Smaller x uses smaller differences, leaving larger differences for later.
            # However, we need to fit k+1 elements.
            # Let's shuffle to explore different branches.
            random.shuffle(candidates)
            
            # Optional: Sort by some heuristic score
            # For now, random shuffle is a good baseline for restart-based search.
            
            for x in candidates:
                # If adding x completes the row, check max constraint
                if len(current_row) == k:
                    if x < prev_max:
                        continue
                
                # Re-check conflicts (safe)
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Commit
                current_row.append(x)
                new_diffs_mask = 0
                for a in current_row[:-1]:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                used |= new_diffs_mask
                
                ok, reason = backtrack(row_idx, current_row, x, prev_max)
                
                if ok:
                    return True, None
                    
                # Backtrack
                current_row.pop()
                used ^= new_diffs_mask
                
                if reason == "timeout":
                    return False, "timeout"
            
            return False, None
        
        # Start backtracking
        # We try multiple times with different random seeds to escape local minima
        # The caller manages the time budget for this scope M.
        
        ok, reason = backtrack(0, [0], 0, -1)
        
        if ok:
            return rows
        else:
            return None

    # Search loop
    # We iterate M from lower_bound to min(fallback_scope, 112+5)
    # If we find a solution, we return it.
    # We allocate a time budget for each M.
    # If M is small (close to LB), it's harder, so maybe less time? Or more?
    # Usually, infeasible small M fail fast. Feasible large M might be easier.
    # Let's allocate a fixed small budget per M, say 2-5 seconds.
    # Total time is 118s.
    
    max_search_scope = min(fallback_scope, 115)
    
    # Sort M values to try? No, we want the smallest M.
    # So we iterate M from lower_bound upwards.
    
    for M in range(lower_bound, max_search_scope + 1):
        if time.time() - start_time > deadline - 2.0:
            break
            
        # Allocate time for this M
        # If M == lower_bound, give it a bit more time?
        # Let's give each M 3 seconds.
        time_budget = 3.0
        
        # Try to solve for this M
        # We can run multiple random restarts within this budget
        # But solve_scope currently does one run.
        # Let's modify the loop to run solve_scope multiple times if time permits.
        
        found = False
        t_start_m = time.time()
        
        while time.time() - t_start_m < time_budget:
            res = solve_scope(M, time_budget - (time.time() - t_start_m))
            if res is not None:
                best_rows = res
                best_scope = M
                found = True
                break
            # If solve_scope returns None, it means it failed or timed out.
            # If it timed out, we should stop trying this M and move to next.
            # If it failed (backtracking exhausted), we should try again with new random seed?
            # solve_scope uses global random state, so it will be different.
            # However, if it exhausts the search tree, it's deterministic failure for that structure.
            # But we don't distinguish between "exhausted" and "timeout" in return value.
            # Let's assume if it returns None quickly, it's exhausted. If slow, timeout.
            # To be safe, if it returns None, we check time. If we used most of budget, stop.
            
            # Simple heuristic: if we just spent a lot of time, break.
            if time.time() - t_start_m > time_budget * 0.8:
                break
            
        if found:
            # We found a solution for M. Since we iterate upwards, this is the best.
            break

    if best_rows is None:
        best_rows = fallback_rows
        best_scope = fallback_scope

    return best_rows

