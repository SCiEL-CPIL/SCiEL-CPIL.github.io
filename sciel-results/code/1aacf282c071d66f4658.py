import time
import random
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin for 150s limit
    
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for scope
    lower_bound = total_diffs
    # Best known upper bound is 134
    # We'll try to go as low as possible.
    
    # Bitmask utilities
    def get_diffs_bitmask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                mask |= 1 << (row[j] - row[i])
        return mask

    def popcount(x):
        return bin(x).count('1')

    def get_valid_candidates(row, used_mask, local_mask, lo, hi, max_scope):
        """
        Returns a list of valid candidates for the next element in [lo, hi].
        Used for backtracking.
        """
        # Precompute forbidden mask based on used_mask and row elements
        # Forbidden x if (x - a) is in used_mask or local_mask for any a in row.
        # (x - a) in mask  <=>  x in mask << a  (shifted mask)
        
        forbidden = 0
        for a in row:
            forbidden |= used_mask << a
            forbidden |= local_mask << a
        
        # We want x in [lo, hi] such that bit x is NOT in forbidden.
        # Also x must be >= lo and <= hi.
        
        candidates = []
        # Iterate x from lo to hi
        # For speed, we can iterate bits, but range is small.
        for x in range(lo, hi + 1):
            if not (forbidden >> x) & 1:
                candidates.append(x)
                
        return candidates

    def backtrack(rows_done, current_row, used_mask, scope, rng):
        """
        Recursive backtracking to fill current_row and then subsequent rows.
        rows_done: list of completed rows
        current_row: list of elements so far in current row
        used_mask: bitmask of all diffs used in completed rows
        scope: current max scope constraint
        rng: random generator
        Returns: list of all rows or None if failed
        """
        if len(rows_done) == n_rows:
            return rows_done
        
        if len(current_row) == m:
            # Row complete
            local_mask = get_diffs_bitmask(current_row)
            # Check overlap with used_mask
            if used_mask & local_mask:
                return None
            
            new_used = used_mask | local_mask
            rows_done.append(current_row)
            
            # Start next row
            next_row = [0]
            result = backtrack(rows_done, next_row, new_used, scope, rng)
            
            if result is not None:
                return result
            else:
                rows_done.pop()
                return None
        
        # Current row not complete, add next element
        last = current_row[-1]
        lo = last + 1
        hi = scope
        
        if lo > hi:
            return None
            
        # Get candidates
        # local_mask is diffs in current_row
        local_mask = get_diffs_bitmask(current_row)
        
        candidates = get_valid_candidates(current_row, used_mask, local_mask, lo, hi, scope)
        
        # Shuffle candidates to break symmetry / allow random restarts?
        # For a deterministic search near the bound, we might want to try smaller ones first.
        # But for finding *a* solution, shuffling helps avoid getting stuck in deep dead ends.
        # Let's sort by value (small to large) to prefer compact solutions?
        # Actually, for backtracking, ordering matters for speed.
        # Smaller values first usually leads to earlier failure or success?
        # Let's try shuffling for randomization, but bias towards smaller?
        # Or just try all in order.
        
        # To make it faster, we can limit the number of candidates we try in this step?
        # No, we must try all to ensure completeness if we want to find a solution.
        
        # Optimization: If candidates is empty, return None.
        if not candidates:
            return None
            
        # Order: Smaller values first (greedy) or Random?
        # Let's try smaller first, as it tends to produce valid solutions if they exist with small scope.
        # But if we get stuck, we need to backtrack.
        
        # To avoid deep recursion timeouts, we can add a timeout check.
        # But Python recursion limit and speed might be an issue.
        # Let's implement a timeout check inside the loop.
        
        # For speed, we'll just iterate.
        # Shuffle to allow different paths?
        # If we always pick smallest, we might hit a dead end that requires a larger earlier choice.
        # Backtracking handles this.
        
        for x in candidates:
            new_row = current_row + [x]
            result = backtrack(rows_done, new_row, used_mask, scope, rng)
            if result is not None:
                return result
        return None

    def backtrack_limited(rows_done, current_row, used_mask, scope, rng, node_limit, time_limit):
        """
        Backtracking with node limit and time limit.
        Returns (rows, nodes_visited) or (None, nodes_visited)
        """
        # We need to pass state for limits. 
        # Simple approach: global counters? No, let's use a wrapper.
        # Actually, for simplicity, let's just use a standard recursive function 
        # but with a check at the top.
        
        # Re-implementing with limit is tricky in pure recursion without passing state.
        # Let's use a class or closure for limits.
        
        # For now, let's assume the search is fast enough or we use a non-recursive stack?
        # Or just a simple counter in a mutable object.
        
        pass # Placeholder

    # Let's implement a simple iterative deepening / bounded search using a stack or just recursion with a global time check.
    
    # Since Python recursion is limited to ~1000, and depth is n_rows * m = 4 * 8 = 32, it's fine.
    # The width is the issue.
    
    # Strategy:
    # 1. Try to find a solution with scope = lower_bound.
    # 2. If failed, increment scope.
    # 3. Use randomization in candidate order to explore different parts of the tree.
    # 4. Keep track of best solution found.
    
    best_solution = None
    best_scope = float('inf')
    
    # Initial greedy solution to have a baseline
    # Use a simple greedy to get a valid solution quickly
    def greedy_construct(scope, rng):
        used_mask = 0
        rows = []
        for i in range(n_rows):
            row = [0]
            local_mask = 0
            for j in range(k):
                lo = row[-1] + 1
                hi = scope
                candidates = []
                for x in range(lo, hi + 1):
                    valid = True
                    # Check against used_mask and local_mask
                    # Forbidden if (x - a) in used_mask or local_mask
                    # We can just compute the check directly
                    for a in row:
                        d = x - a
                        if (used_mask >> d) & 1 or (local_mask >> d) & 1:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        break # Greedy: take first valid (smallest)
                
                if not candidates:
                    return None
                
                x = candidates[0]
                row.append(x)
                # Update local_mask
                for a in row[:-1]:
                    local_mask |= 1 << (x - a)
            
            # Check final row validity (duplicates within row already prevented by local_mask check? 
            # Wait, the check above prevents d in local_mask. So no duplicates in row.)
            # Check against global used_mask
            if used_mask & local_mask:
                return None
            
            rows.append(row)
            used_mask |= local_mask
            
        return rows
    
    # Get an initial solution with a generous scope
    init_scope = 200
    rows = greedy_construct(init_scope, random.Random(42))
    if rows:
        best_solution = rows
        best_scope = max(max(r) for r in rows)
    
    # Now, try to improve using backtracking with increasing scope from lower_bound
    # We can also try to reduce the scope of the current best solution using local search?
    # Or just search for the minimum scope using backtracking.
    
    # Search strategy:
    # For scope in range(lower_bound, best_scope + 1):
    #    Try to find a solution with this scope.
    #    If found, update best.
    #    If time runs out, break.
    
    # To make backtracking effective, we need to prune aggressively.
    # The bitmask check is O(1) for validity of a candidate given the masks.
    # But generating candidates is O(k * scope).
    
    # Let's implement a backtracking that tries to fill rows.
    # We will use a global time check to abort long searches.
    
    search_start = time.time()
    
    def search_scope(target_scope, rng):
        """
        Try to find a valid solution with max entry <= target_scope.
        Uses randomized backtracking.
        Returns solution or None.
        """
        used_mask = 0
        rows_done = []
        current_row = [0]
        local_mask = 0
        
        # Recursion depth is small.
        # We will use a stack-based approach or just recursion.
        # Recursion is cleaner.
        
        def dfs(row_idx, pos_in_row, row, used_mask, local_mask):
            # Check time
            if time.time() > search_start + 140:
                return None
            
            if row_idx == n_rows:
                return rows_done.copy()
            
            if pos_in_row == m:
                # Row complete
                if used_mask & local_mask:
                    return None
                
                rows_done.append(row)
                res = dfs(row_idx + 1, 0, [0], used_mask | local_mask, 0)
                if res is not None:
                    return res
                rows_done.pop()
                return None
            
            # Add next element
            last = row[-1]
            lo = last + 1
            hi = target_scope
            
            if lo > hi:
                return None
            
            # Find candidates
            candidates = []
            for x in range(lo, hi + 1):
                valid = True
                for a in row:
                    d = x - a
                    if (used_mask >> d) & 1 or (local_mask >> d) & 1:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Randomize candidate order to allow exploration
            # But bias towards smaller?
            # Let's shuffle.
            rng.shuffle(candidates)
            
            for x in candidates:
                new_row = row + [x]
                new_local_mask = local_mask
                for a in row:
                    new_local_mask |= 1 << (x - a)
                
                res = dfs(row_idx, pos_in_row + 1, new_row, used_mask, new_local_mask)
                if res is not None:
                    return res
            
            return None
        
        return dfs(0, 1, [0], 0, 0)
    
    # Iterate scopes from lower_bound up to current best
    # We can also randomize the starting scope or use a jump?
    # Lower bound is tight. Let's start from there.
    
    found_better = False
    for scope in range(lower_bound, best_scope + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts for this scope
        for attempt in range(20):  # Limit restarts to not waste time
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(attempt * 1000 + scope)
            sol = search_scope(scope, rng)
            if sol:
                # Verify
                used = set()
                valid = True
                for r in sol:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in used:
                                valid = False
                                break
                            used.add(d)
                    if not valid:
                        break
                if valid:
                    best_solution = sol
                    best_scope = scope
                    found_better = True
                    break # Found solution for this scope, try next lower scope? 
                          # No, we are iterating UP. If we found one at scope, 
                          # we want to see if a LOWER scope works. 
                          # But we started at lower_bound. So if we found one at lower_bound, we are done.
                          # If we found one at scope > lower_bound, we can stop searching higher?
                          # No, we want the MINIMUM. So if we start at lower_bound and go up, 
                          # the first success is the best.
            
            # If we didn't find a solution for this scope, we break the inner loop and go to next scope?
            # Yes, because if no solution exists at this scope, none will be found with more restarts (usually).
            # But maybe a bad random order prevented finding it?
            # So we break the inner loop and increment scope.
            break # Move to next scope
            
    # If no better solution found via backtracking, keep the greedy one.
    
    # Final polish: Local search on the best solution found
    # Try to reduce the max element by moving elements left.
    if best_solution:
        current_sol = [r[:] for r in best_solution]
        
        # Compute used diffs for current sol
        def get_all_diffs(sol):
            used = set()
            for r in sol:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        used.add(r[j] - r[i])
            return used
        
        # Iterative local search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for row_idx in range(n_rows):
                row = current_sol[row_idx]
                # Try to move each element (from last to first) to a smaller value
                for pos in range(k, 0, -1):
                    old_val = row[pos]
                    if old_val == 0:
                        continue
                    # Try new values from row[pos-1]+1 to old_val-1
                    # We need to check validity.
                    # To speed up, we can precompute the set of forbidden diffs for this position?
                    # Forbidden diffs are all diffs in the solution EXCEPT those involving row[pos]?
                    # No, if we change row[pos], the diffs involving row[pos] change.
                    # The other diffs remain.
                    
                    # Let's collect all other diffs
                    other_diffs = set()
                    for r in current_sol:
                        if r is row:
                            # Diffs in this row NOT involving pos
                            for i in range(len(r)):
                                for j in range(i+1, len(r)):
                                    if i != pos and j != pos:
                                        other_diffs.add(r[j] - r[i])
                        else:
                            for i in range(len(r)):
                                for j in range(i+1, len(r)):
                                    other_diffs.add(r[j] - r[i])
                    
                    # Also, within the row, diffs involving pos are removed and new ones added.
                    # The new diffs must not be in other_diffs.
                    # And the new diffs must be distinct from each other (within the new set involving pos).
                    
                    found_better_val = False
                    for new_val in range(row[pos-1] + 1, old_val):
                        # Check if new_val creates duplicate diffs
                        new_diffs = []
                        valid = True
                        for i in range(pos):
                            d = new_val - row[i]
                            if d in other_diffs:
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if not valid:
                            continue
                        
                        # Check for duplicates within new_diffs
                        if len(set(new_diffs)) != len(new_diffs):
                            continue
                        
                        # Also need to check that new_diffs don't conflict with each other?
                        # They are just a set, so distinctness is checked above.
                        
                        # If valid, apply
                        row[pos] = new_val
                        improved = True
                        found_better_val = True
                        break
                    
                    if found_better_val:
                        # Update current_sol and re-compute other_diffs for next iteration?
                        # The loop continues, so we can just continue.
                        # But `other_diffs` is stale.
                        # We can break out of the inner loops and restart?
                        # Or just continue, knowing it might be slightly inefficient.
                        # Given small size, it's fine.
                        pass
                    
        # Update best if improved
        if improved:
            # Verify
            used = set()
            valid = True
            for r in current_sol:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in used:
                            valid = False
                            break
                        used.add(d)
                if not valid:
                    break
            if valid:
                best_solution = current_sol

    return best_solution

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    if sol:
        print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
    else:
        print("No solution found")
