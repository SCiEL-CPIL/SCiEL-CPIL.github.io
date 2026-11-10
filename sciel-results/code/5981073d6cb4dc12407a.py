import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # 1. Greedy Baseline
    def greedy_solution():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                found = False
                for x in range(last + 1, 1000):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        last = x
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    baseline_rows = greedy_solution()
    if baseline_rows is None:
        # Fallback for very small k or errors
        return [[i for i in range(k+1)] for _ in range(n_rows)]

    best_m = max(max(r) for r in baseline_rows)
    best_solution = baseline_rows

    # 2. Backtracking Solver to improve
    # We try to see if M-1, M-2, etc., are feasible.
    
    def try_construct_for_M(M):
        """
        Returns a valid list of rows if a solution exists with max <= M, else None.
        Uses backtracking with symmetry breaking.
        """
        used_diffs = set()
        rows = []
        row_size = k + 1
        
        # Time limit for this specific search
        # We want to leave time for other attempts
        local_deadline = start_time + (deadline - start_time) - 10 
        
        def backtrack(row_idx, current_row, used_diffs, prev_row):
            if time.time() > local_deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            # If current row is full
            if len(current_row) == row_size:
                # Symmetry check: current_row >= prev_row
                if row_idx > 0 and current_row < prev_row:
                    return False
                
                rows.append(current_row.copy())
                
                # Recurse to next row
                if backtrack(row_idx + 1, [], used_diffs, current_row):
                    return True
                
                rows.pop()
                return False
            
            # Extend current row
            last_val = current_row[-1]
            
            # Heuristic: Try smaller values first
            for x in range(last_val + 1, M + 1):
                # Check if x can be added
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Prune using symmetry prefix
                if row_idx > 0:
                    # Check if adding x makes the prefix invalid
                    # We compare current_row + [x] with prev_row
                    bad = False
                    # Only need to check up to the new length
                    new_len = len(current_row) + 1
                    for i in range(new_len):
                        val = current_row[i] if i < len(current_row) else x
                        if val < prev_row[i]:
                            bad = True
                            break
                        if val > prev_row[i]:
                            break # Safe
                    
                    if bad:
                        continue
                
                # Commit
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row, used_diffs, prev_row):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        # Start
        if backtrack(0, [0], used_diffs, None):
            return rows
        return None

    # Try to improve
    # We know best_m works. Try best_m - 1.
    # If it succeeds, try best_m - 2, etc.
    # We stop if we fail (and didn't timeout) or if time runs out.
    
    # To distinguish timeout from infeasible, we can check if time.time() < local_deadline
    # But local_deadline is local to try_construct_for_M.
    # Let's just try a few steps.
    
    for m in range(best_m - 1, max(100, best_m - 20), -1):
        if time.time() > deadline - 15:
            break
        
        sol = try_construct_for_M(m)
        if sol is not None:
            best_m = m
            best_solution = sol
            # Continue to try to lower it further
        else:
            # If it returned None, it might be infeasible or timed out.
            # If it timed out, we shouldn't assume infeasible.
            # However, if we just spent 10s on it, we have little time left.
            # And if it's infeasible, smaller M will also be infeasible.
            # So breaking is safe in both cases (we keep the current best_solution).
            break

    return best_solution
