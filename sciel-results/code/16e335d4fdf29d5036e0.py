import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 110.0  # Safety margin
    row_len = k + 1
    
    # For n_rows=7, k=5, row_len=6
    # Differences range from 1 to max_scope
    # We use a bitmask for differences. 
    # Max difference is max_scope. 
    # If max_scope is ~112, we need 112 bits. Python ints handle this easily.
    
    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                m |= (1 << d)
        return m

    def is_valid_full(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def build_row_dfs(used_mask, max_scope, rng=None):
        """
        Finds a valid row of length row_len, starting with 0, 
        with last element <= max_scope, using differences not in used_mask.
        Prioritizes smaller last elements.
        Returns the row or None.
        """
        best_row = [None]
        
        def dfs(row, local_mask, last_val):
            if len(row) == row_len:
                if best_row[0] is None or row[-1] < best_row[0][-1]:
                    best_row[0] = row[:]
                return
            
            # If we've already found a row, we only care about finding a better one (smaller last val)
            # But for the first find, we just want any valid one.
            # To speed up, if we find one, we can stop if we don't need the absolute minimum for this specific call,
            # but finding the minimum helps keep global scope low.
            
            # Optimization: If we have a best_row, we can prune if last_val + (row_len - len(row)) > best_row[0][-1]
            # assuming we can add at least 1 per step.
            
            start = last_val + 1
            # Upper bound for this step: 
            # We need to fit (row_len - len(row) - 1) more elements after this one.
            # So this element cannot be > max_scope - (row_len - len(row) - 1)
            end = max_scope - (row_len - len(row) - 1)
            
            if start > end:
                return
            
            # Try values in increasing order to find the tightest row
            for x in range(start, end + 1):
                # Check conflicts with used_mask and local_mask
                new_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if (used_mask & bit) or (local_mask & bit):
                        valid = False
                        break
                    new_mask |= bit
                if not valid:
                    continue
                
                row.append(x)
                dfs(row, local_mask | new_mask, x)
                row.pop()
                
                # If we found a very good row (e.g., last element is minimal possible), we can stop
                # The minimal possible last element for a row of length L starting at 0 is L-1 (1,2,3..L-1)
                # But differences must be distinct.
                # For row [0, 1, 2, 3, 4, 5], diffs are 1,2,3,4,5,1,2,3,1,2,1. Collisions!
                # So the minimal last element is higher.
                if best_row[0] is not None:
                    # If we found a row, and we are trying to minimize, 
                    # and the current x is already >= best_row[0][-1], we can break?
                    # No, because a smaller x might lead to a valid completion with a smaller final element?
                    # Actually, if x is larger, the final element will likely be larger.
                    # So if x >= best_row[0][-1], we can break.
                    if x >= best_row[0][-1]:
                        break

        dfs([0], 0, 0)
        return best_row[0]

    def solve_with_scope(max_scope, time_limit=10.0):
        """
        Attempts to find a valid set of n_rows with scope <= max_scope.
        Uses backtracking at the row level.
        """
        start_time = time.time()
        rows = []
        used_mask = 0
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            
            if idx == n_rows:
                return True
            
            # Generate a candidate row
            # To allow backtracking, we might need to try multiple rows for this index.
            # For now, we try the "best" (tightest) row first.
            # If that fails, we need a way to get the *next* best row.
            # This is complex. Instead, we can randomize the choice slightly or use a list of candidates.
            
            # Simple approach: Try to build the tightest row.
            row = build_row_dfs(used_mask, max_scope)
            
            if row is None:
                return False
            
            # If we have multiple options, we should try them.
            # For now, let's assume the tightest is best.
            # If this leads to a dead end, we would need to backtrack and try a different row for this idx.
            # To do that, build_row_dfs would need to yield rows.
            
            # Let's modify build_row_dfs to yield rows in order of last element.
            # For simplicity in this rewrite, I will implement a generator version.
            
            return False # Placeholder, will be replaced by generator logic

    # Refactored approach: Generator for rows
    
    def row_generator(used_mask, max_scope):
        """
        Yields valid rows for the given used_mask and max_scope, 
        ordered by last element (ascending).
        """
        found_rows = []
        
        def dfs(row, local_mask, last_val):
            if len(row) == row_len:
                found_rows.append(row[:])
                return
            
            start = last_val + 1
            end = max_scope - (row_len - len(row) - 1)
            
            if start > end:
                return
            
            for x in range(start, end + 1):
                new_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if (used_mask & bit) or (local_mask & bit):
                        valid = False
                        break
                    new_mask |= bit
                if not valid:
                    continue
                
                row.append(x)
                dfs(row, local_mask | new_mask, x)
                row.pop()
                
                # Limit the number of rows found to prevent explosion
                if len(found_rows) > 50:
                    return

        dfs([0], 0, 0)
        
        # Sort by last element
        found_rows.sort(key=lambda r: r[-1])
        
        for r in found_rows:
            yield r

    def solve_backtracking(max_scope, time_limit=5.0):
        start_time = time.time()
        rows = []
        used_mask = 0
        solutions_found = []
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            
            if idx == n_rows:
                solutions_found.append([list(r) for r in rows])
                return True # Stop at first solution found? Or find all?
                # Finding the first solution with minimal scope is good.
            
            # Get candidates for this row
            # We want to try candidates with smallest last element first
            candidates = list(row_generator(used_mask, max_scope))
            
            if not candidates:
                return False
            
            # Try each candidate
            for row in candidates:
                rows.append(row)
                new_mask = row_mask(row)
                
                if backtrack(idx + 1):
                    return True
                
                rows.pop()
                
                # If we found a solution, we return. 
                # If we want to find a better one (smaller scope), we handle that in the outer loop.
                
            return False
        
        success = backtrack(0)
        if success and solutions_found:
            return solutions_found[0]
        return None

    # Main logic
    best_solution = None
    best_scope = float('inf')
    
    # Try scopes from lower bound upwards
    # Counting bound is 105. Try 105, 106, ...
    # If we find one at 105, it's likely optimal.
    
    # First, get a quick upper bound using a simple greedy to avoid starting too high
    # But we know 112 is a known solution. Let's try 105 to 112.
    
    for scope in range(105, 113):
        if time.time() > DEADLINE - 10:
            break
            
        # Try to solve for this scope
        # Use a short time limit per scope to ensure we try multiple scopes
        sol = solve_backtracking(scope, time_limit=5.0)
        
        if sol is not None:
            best_solution = sol
            best_scope = scope
            # If we found a solution at the lower bound (105), it's optimal.
            if scope == 105:
                break
        else:
            # If no solution found, try the next scope
            pass

    # If no solution found in the range 105-112, use a fallback greedy
    if best_solution is None:
        # Fallback: Greedy construction
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = build_row_dfs(used_mask, 200) # High limit
            if row is None:
                # Should not happen
                break
            rows.append(row)
            used_mask |= row_mask(row)
        
        if rows:
            best_solution = rows
            best_scope = max(r[-1] for r in rows)
        else:
            # Absolute fallback
            rows = []
            for _ in range(n_rows):
                row = [0]
                for i in range(1, row_len):
                    v = row[-1] + 1
                    while True:
                        valid = True
                        for prev in row:
                            d = v - prev
                            if (used_mask >> d) & 1:
                                valid = False
                                break
                        if valid:
                            row.append(v)
                            for prev in row[:-1]:
                                used_mask |= (1 << (v - prev))
                            break
                        v += 1
                rows.append(row)
            best_solution = rows
            best_scope = max(r[-1] for r in rows)

    # Final check and return
    # The problem asks for a list of rows.
    return best_solution

