"""
Improved program for Difference Triangle Set (4, 7).
Uses Randomized Backtracking Search with pruning and local polishing.
"""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    # k is typically 7 for this task, meaning rows have k+1 = 8 elements.
    # n_rows is typically 4.
    # The task description says "4 rows of 8 increasing integers".
    # So k=7, n_rows=4.
    
    # If parameters are different, we adapt, but the logic holds.
    row_len = k + 1
    
    # 1. Greedy Baseline
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < row_len:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(row) for row in best_rows)
    
    # 2. Helper functions
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in get_row_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    # 3. Backtracking Search
    # We try to build rows one by one.
    # To make it effective, we use a randomized order for the "first" few elements or rely on the fact that 
    # constructing a tight row first restricts the space for subsequent rows.
    
    def backtrack_search(timeout):
        """
        Attempts to find a solution with a smaller scope than best_scope.
        Returns (rows, scope) or (None, None).
        """
        nonlocal best_rows, best_scope
        
        # We will try to construct n_rows rows.
        # State: rows constructed so far, set of used differences.
        
        # Heuristic: Sort potential "start" patterns? 
        # For now, we do a DFS where we pick the next element for the current row.
        
        # To improve, we can fix the first row to be the "greedy" tight row, 
        # as it is often part of the optimal solution or close to it.
        # Let's try fixing the first row to the greedy row first, then searching for the rest.
        # Then we try randomizing the first row.
        
        def dfs(rows_so_far, used_diffs, current_row_idx, current_row, current_max_allowed):
            nonlocal best_rows, best_scope
            
            if time_left() < 1.0:
                return False # Timeout, abort this branch

            if current_row_idx == n_rows:
                # All rows constructed
                scope = max(max(r) for r in rows_so_far)
                if scope < best_scope:
                    best_rows = [list(r) for r in rows_so_far]
                    best_scope = scope
                    # We don't return True to stop immediately because we might find a better one,
                    # but for this simple DFS, we can return True to signal success if we want to stop early.
                    # However, we want to explore more if possible.
                    # Let's just record it and continue searching for even smaller scope.
                    # Actually, if we find a scope, we can prune branches that can't beat it.
                    return True
            
            # Pruning: If we are constructing a row, and the current max is already >= best_scope, 
            # we might still continue if we are in the last row and can reduce it? No, max is non-decreasing.
            # If current_row is not empty and max(current_row) >= best_scope, we can prune this branch 
            # unless we are hoping to reduce previous rows? No, previous rows are fixed in this DFS path.
            # So if any row in rows_so_far has max >= best_scope, we can prune?
            # Actually, best_scope is the global best. If current partial solution already has a row with max >= best_scope,
            # it cannot lead to a solution with scope < best_scope.
            if any(max(r) >= best_scope for r in rows_so_far):
                return False
            
            if current_row_idx < len(rows_so_far):
                # This case shouldn't happen if we manage state correctly
                pass

            # We are constructing row at index current_row_idx
            # current_row is the list of elements so far for this row.
            
            if len(current_row) == row_len:
                # Row complete
                # Check if this row's diffs are valid with used_diffs
                # (This is checked during construction, but double check just in case)
                row_diffs = get_row_diffs(current_row)
                if row_diffs & used_diffs:
                    return False
                
                # Recurse to next row
                return dfs(rows_so_far + [current_row], used_diffs | row_diffs, current_row_idx + 1, [], best_scope)
            
            # Determine the next element for current_row
            # The next element must be > current_row[-1]
            # And its differences with all previous elements in current_row must be:
            # 1. Not in used_diffs
            # 2. Not in current_row's existing diffs
            
            last_val = current_row[-1]
            current_row_diffs = get_row_diffs(current_row)
            
            # Heuristic: Try values in increasing order, but with some randomization to escape local minima?
            # For backtracking, increasing order is usually best for finding *a* solution.
            # To find a *better* solution, we might need to explore different structures.
            # However, since we prune based on best_scope, increasing order helps find tight solutions first.
            
            # Range to try: from last_val + 1 to best_scope - 1 (optimistic bound)
            # We can also use a tighter bound based on how many elements are left.
            # If we need 'm' more elements, the minimum additional span is roughly m*(m+1)/2? No, just m.
            # So max_val <= best_scope - (row_len - len(current_row))
            
            min_next = last_val + 1
            # Optimistic max next value
            remaining_elements = row_len - len(current_row)
            max_next = best_scope - remaining_elements
            
            if min_next > max_next:
                return False

            # To add diversity, we can shuffle the order of trials slightly, 
            # or use a "lookahead" to avoid values that block future elements.
            # For now, simple increasing order with a cap.
            
            # Optimization: Pre-calculate forbidden diffs for this step
            # forbidden = used_diffs | current_row_diffs
            
            for val in range(min_next, max_next + 1):
                if time_left() < 1.0:
                    return False
                
                # Check differences
                new_diffs = {val - a for a in current_row}
                if new_diffs & used_diffs or new_diffs & current_row_diffs:
                    continue
                
                # Valid candidate
                # Recurse
                if dfs(rows_so_far, used_diffs, current_row_idx, current_row + [val], best_scope):
                    # If we found a better solution, we could return True to stop this branch,
                    # but we want to keep searching for even better ones in other branches?
                    # Actually, if we found a solution with scope S, any branch that can't beat S is pruned.
                    # Returning True allows the caller to potentially prune other branches if they are worse.
                    # But here, the "scope" is determined at the end. 
                    # Let's just return True to indicate "success found in this subtree" 
                    # which might allow higher levels to prune if they have a "current best" logic.
                    # In this simple DFS, we just record the global best.
                    return True
            
            return False

        # Start the search
        # We will try a few different starting strategies.
        # Strategy 1: Fix first row to greedy.
        greedy_row = best_rows[0]
        greedy_row_diffs = get_row_diffs(greedy_row)
        
        # Run DFS with first row fixed
        # We need to handle the case where we start with 0 rows or 1 row.
        # Let's modify dfs to accept a starting state.
        
        def dfs_with_start(start_rows, start_used_diffs, start_row_idx, start_row, current_max_allowed):
            nonlocal best_rows, best_scope
            
            if time_left() < 1.0:
                return False

            if start_row_idx == n_rows:
                scope = max(max(r) for r in start_rows)
                if scope < best_scope:
                    best_rows = [list(r) for r in start_rows]
                    best_scope = scope
                return True
            
            # Prune
            if any(max(r) >= best_scope for r in start_rows):
                return False
            
            if len(start_row) == row_len:
                row_diffs = get_row_diffs(start_row)
                if row_diffs & start_used_diffs:
                    return False
                return dfs_with_start(start_rows + [start_row], start_used_diffs | row_diffs, start_row_idx + 1, [], current_max_allowed)
            
            last_val = start_row[-1]
            current_row_diffs = get_row_diffs(start_row)
            
            min_next = last_val + 1
            remaining_elements = row_len - len(start_row)
            max_next = current_max_allowed - remaining_elements
            
            if min_next > max_next:
                return False

            # Randomize the order of trials slightly to explore different structures?
            # Or just increasing. Increasing is good for finding tight rows.
            # Let's stick to increasing for now, as it's more systematic.
            
            for val in range(min_next, max_next + 1):
                if time_left() < 1.0:
                    return False
                
                new_diffs = {val - a for a in start_row}
                if new_diffs & start_used_diffs or new_diffs & current_row_diffs:
                    continue
                
                if dfs_with_start(start_rows, start_used_diffs, start_row_idx, start_row + [val], current_max_allowed):
                    return True
            
            return False

        # Attempt 1: Fix first row to greedy
        if time_left() > 5:
            dfs_with_start([greedy_row], greedy_row_diffs, 1, [], best_scope)
            
        # Attempt 2: Randomized first row
        # We can generate a few random tight first rows and use them as seeds.
        for _ in range(50):
            if time_left() < 5:
                break
            
            # Generate a random tight row
            row = [0]
            row_diffs = set()
            x = 0
            success = True
            # Use a randomized greedy to generate the first row
            # This allows the first row to vary, which might open up better solutions for subsequent rows.
            while len(row) < row_len:
                found = False
                # Look ahead
                for trial_x in range(x + 1, x + 20):
                    new = {trial_x - a for a in row}
                    if not (new & row_diffs):
                        # Add some randomness
                        if random.random() < 0.1:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            
            if success:
                r_scope = max(row)
                if r_scope < best_scope:
                    dfs_with_start([row], row_diffs, 1, [], best_scope)

        return best_rows, best_scope

    # 4. Local Polish (Refinement)
    def local_polish(rows):
        """
        Try to reduce the scope of each row individually, keeping others fixed.
        """
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 5
        pass_count = 0
        
        while improved and time_left() > 2 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Process rows in order of largest max
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 2:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= get_row_diffs(rows[j])
                
                # Try to rebuild row i to be smaller
                current_row = rows[i]
                current_max = max(current_row)
                
                # Backtracking to find a smaller row
                def find_smaller_row(target_max):
                    # Returns a row with max <= target_max or None
                    # Uses DFS
                    def dfs_row(row, used_local, target):
                        if len(row) == row_len:
                            if max(row) <= target:
                                return row
                            return None
                        last = row[-1]
                        # Prune: if last + (row_len - len(row)) > target, impossible
                        if last + (row_len - len(row)) > target:
                            return None
                        for val in range(last + 1, target + 1):
                            new_diffs = {val - a for a in row}
                            if new_diffs & used_local or new_diffs & used_other:
                                continue
                            res = dfs_row(row + [val], used_local | new_diffs, target)
                            if res:
                                return res
                        return None
                    
                    return dfs_row([0], set(), target)

                # Binary search for the smallest possible max for this row?
                # Or just try current_max - 1, current_max - 2, etc.
                # Since we want to minimize, try to reduce.
                new_row = None
                for trial_max in range(current_max - 1, 0, -1):
                    if time_left() < 2:
                        break
                    candidate = find_smaller_row(trial_max)
                    if candidate:
                        new_row = candidate
                        break
                
                if new_row:
                    rows[i] = new_row
                    improved = True
        
        return rows

    # Main Loop
    # 1. Greedy
    # 2. Backtracking Search
    # 3. Local Polish
    # 4. Repeat Backtracking/Polish if time allows

    # Run Backtracking
    if time_left() > 10:
        backtrack_search()
        
    # Polish the result of backtracking
    if time_left() > 5:
        polished = local_polish(best_rows)
        if max(max(r) for r in polished) < best_scope:
            best_rows = polished
            best_scope = max(max(r) for r in best_rows)
            
    # One more pass of backtracking with the new best_scope as bound
    if time_left() > 10:
        backtrack_search()
        
    # Final Polish
    if time_left() > 2:
        polished = local_polish(best_rows)
        if max(max(r) for r in polished) < best_scope:
            best_rows = polished
            best_scope = max(max(r) for r in best_rows)
            
    # Final Validity Check
    if not is_valid(best_rows):
        best_rows = greedy()
        
    return best_rows
