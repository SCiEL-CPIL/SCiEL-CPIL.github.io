<<<<<<< SEARCH
    # ---------- Targeted Backtracking Search ----------
    def backtrack_search(target_scope, time_limit):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
        Returns rows if found, None otherwise.
        """
        start = time.time()
        
        # Pre-compute all possible diffs up to target_scope
        # We need to fill n_rows * C(k+1, 2) distinct diffs.
        # For n_rows=4, k=7, C(8,2)=28. Total 112 diffs.
        # If target_scope < 112, it's impossible by counting bound.
        
        # State:
        # rows: list of lists
        # used_diffs: set of used diffs
        
        rows = []
        used_diffs = set()
        
        # We build rows one by one.
        # Within a row, we build elements one by one.
        
        def backtrack_row(row_idx, current_row, row_diffs):
            if time.time() - start > time_limit:
                return False
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                used_diffs.update(row_diffs)
                
                if row_idx == n_rows - 1:
                    # All rows complete
                    return True
                
                return backtrack_row(row_idx + 1, [0], set())
            
            # Determine next element
            # It must be > current_row[-1] and <= target_scope
            last_val = current_row[-1]
            
            # Candidate values
            # To make it randomized, we can shuffle the range or pick randomly from valids
            # But we need to check validity.
            
            # Optimization: Only try values that don't immediately conflict?
            # For backtracking, we try candidates.
            
            # Range of candidates: last_val + 1 to target_scope
            # To speed up, we can limit the search space slightly or use heuristics.
            # Let's try a subset of candidates first, then expand if stuck.
            # For simplicity in this implementation, we iterate but with a time check.
            
            # To avoid deep recursion overhead and allow randomization, we can use a stack or recursion.
            # Recursion is fine for depth k+1 (8).
            
            # Heuristic: Try smaller values first? Or random?
            # Random helps escape traps.
            # Let's try a random subset of candidates first.
            
            candidates = list(range(last_val + 1, target_scope + 1))
            random.shuffle(candidates)
            
            # Limit candidates to avoid huge branching factor if target_scope is large
            # But target_scope should be close to best_scope, so it's manageable.
            
            for val in candidates:
                if time.time() - start > time_limit:
                    return False
                
                # Check diffs
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = val - a
                    if d in used_diffs or d in row_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Recurse
                    if backtrack_row(row_idx, current_row + [val], row_diffs | new_diffs):
                        return True
                    # Backtrack
                    # (No explicit cleanup needed as we use immutable structures in recursion mostly, 
                    # but current_row is a list. We pass a new list current_row + [val])
            
            return False

        # Start backtracking
        # We need to be careful with the 'rows' and 'used_diffs' being mutated.
        # Let's use a cleaner recursive approach with local state or explicit stack.
        
        def solve_bt():
            rows_local = []
            used_local = set()
            
            def rec(row_idx, curr_row, curr_diffs):
                if time.time() - start > time_limit:
                    return False
                
                if len(curr_row) == k + 1:
                    rows_local.append(curr_row)
                    used_local.update(curr_diffs)
                    
                    if row_idx == n_rows - 1:
                        return True
                    
                    return rec(row_idx + 1, [0], set())
                
                last_val = curr_row[-1]
                candidates = list(range(last_val + 1, target_scope + 1))
                random.shuffle(candidates)
                
                for val in candidates:
                    if time.time() - start > time_limit:
                        return False
                    
                    new_diffs = set()
                    valid = True
                    for a in curr_row:
                        d = val - a
                        if d in used_local or d in curr_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        if rec(row_idx, curr_row + [val], curr_diffs | new_diffs):
                            return True
                
                return False
            
            if rec(0, [0], set()):
                return rows_local
            return None

        return solve_bt()
=======
    # ---------- Targeted Backtracking Search ----------
    def backtrack_search(target_scope, time_limit):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking with bitmasks.
        Returns rows if found, None otherwise.
        """
        start = time.time()
        
        def solve_bt():
            rows_local = []
            used_local = 0
            
            def rec(row_idx, curr_row, curr_mask):
                if time.time() - start > time_limit:
                    return False
                
                if len(curr_row) == k + 1:
                    rows_local.append(curr_row)
                    used_local |= curr_mask
                    
                    if row_idx == n_rows - 1:
                        return True
                    
                    return rec(row_idx + 1, [0], 0)
                
                last_val = curr_row[-1]
                candidates = list(range(last_val + 1, target_scope + 1))
                random.shuffle(candidates)
                
                for val in candidates:
                    if time.time() - start > time_limit:
                        return False
                    
                    new_mask = 0
                    valid = True
                    for a in curr_row:
                        d = val - a
                        bit = 1 << d
                        if (used_local & bit) or (curr_mask & bit):
                            valid = False
                            break
                        new_mask |= bit
                    
                    if valid:
                        if rec(row_idx, curr_row + [val], curr_mask | new_mask):
                            return True
                
                return False
            
            if rec(0, [0], 0):
                return rows_local
            return None

        return solve_bt()
>>>>>>> REPLACE

