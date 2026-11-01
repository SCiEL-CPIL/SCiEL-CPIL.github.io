import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    M_lb = total_diffs_needed
    
    # Greedy construction for initial upper bound
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows
    
    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    # Check if we already have a good solution
    # For (4,7), greedy typically gives scope around 200+
    # We want to find something closer to 134
    
    # Strategy: Use a constraint-based search with forward checking
    # but with a twist: we'll use a "row-at-a-time" approach where
    # each row is constructed to minimize its own span while respecting
    # the global difference constraints.
    
    # Key insight: For small n_rows and k, we can use a smarter
    # row construction that considers the global state more carefully.
    
    def try_row_construction(used_diffs, M, target_row_idx):
        """
        Try to construct a single row that fits within scope M
        and doesn't conflict with used_diffs.
        Returns the row list if successful, None otherwise.
        """
        row = [0]
        
        def backtrack_row(c):
            if time.time() > deadline - 2:
                return None
            
            if c == k + 1:
                return list(row)
            
            lower = row[c-1] + 1
            # Upper bound: must leave room for remaining elements
            remaining = k - c
            upper = min(M, M - remaining)
            
            if lower > upper:
                return None
            
            # Try values, preferring those that create "useful" differences
            # (ones that are smaller, to leave more room for other rows)
            candidates = []
            for val in range(lower, upper + 1):
                # Quick check against used_diffs
                valid = True
                for j in range(c):
                    d = val - row[j]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
            
            if not candidates:
                return None
            
            # Heuristic: try values that minimize the maximum difference
            # created with previous elements in this row
            # But also consider that we need to leave differences available
            # for other rows, so prefer "smaller" differences
            candidates.sort()
            
            for val in candidates:
                row.append(val)
                new_diffs = set(val - row[j] for j in range(c))
                
                # Check against global used_diffs
                if not new_diffs & used_diffs:
                    # Tentatively add
                    old_len = len(used_diffs)
                    used_diffs.update(new_diffs)
                    
                    result = backtrack_row(c + 1)
                    if result is not None:
                        return result
                    
                    # Undo
                    used_diffs.clear()
                    # Re-add original used_diffs - this is inefficient!
                    # Let's use a different approach with a set copy
            
            row.pop()
            return None
        
        return backtrack_row(1)
    
    # The above approach has a problem: modifying used_diffs in place
    # and then restoring it is inefficient. Let's use a different strategy.
    
    # Alternative: Use a complete backtracking solver but with better heuristics
    # and a time-aware approach.
    
    def solve_with_backtracking(M, time_budget):
        """
        Complete backtracking solver with forward checking.
        Returns rows if solution found, None otherwise.
        """
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        def backtrack(r, c):
            if time.time() > time_budget:
                return False
            
            if r == n_rows:
                return True
            
            if c == k + 1:
                return backtrack(r + 1, 1)
            
            lower = rows[r][c-1] + 1
            upper = min(M, M - (k - c))
            
            if lower > upper:
                return False
            
            # Try values in increasing order (prefer smaller to minimize span)
            for val in range(lower, upper + 1):
                # Check against all previously assigned elements
                valid = True
                
                # Same row, previous columns
                for j in range(c):
                    d = val - rows[r][j]
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Previous rows
                for i in range(r):
                    for j in range(k + 1):
                        d = abs(val - rows[i][j])
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Commit
                rows[r][c] = val
                new_diffs = set()
                for j in range(c):
                    new_diffs.add(val - rows[r][j])
                for i in range(r):
                    for j in range(k + 1):
                        new_diffs.add(abs(val - rows[i][j]))
                
                used_diffs.update(new_diffs)
                
                if backtrack(r, c + 1):
                    return True
                
                # Uncommit
                used_diffs.difference_update(new_diffs)
                rows[r][c] = 0
            
            return False
        
        return backtrack(0, 1)
    
    # Try to find a solution with scope less than the greedy upper bound
    # Start from M_lb and increase
    
    # To improve efficiency, we can use randomization in the value ordering
    # to escape bad local branches
    
    def solve_with_randomized_backtracking(M, time_budget):
        """
        Backtracking with randomized value ordering to improve search efficiency.
        """
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        def backtrack(r, c):
            if time.time() > time_budget:
                return False
            
            if r == n_rows:
                return True
            
            if c == k + 1:
                return backtrack(r + 1, 1)
            
            lower = rows[r][c-1] + 1
            upper = min(M, M - (k - c))
            
            if lower > upper:
                return False
            
            # Generate candidates and shuffle for randomization
            candidates = list(range(lower, upper + 1))
            random.shuffle(candidates)
            
            # But bias towards smaller values (they're more likely to lead to lower scope)
            # Use a weighted shuffle: sort by a mix of value and randomness
            candidates.sort(key=lambda v: v + random.random() * 10)
            
            for val in candidates:
                valid = True
                
                for j in range(c):
                    d = val - rows[r][j]
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                for i in range(r):
                    for j in range(k + 1):
                        d = abs(val - rows[i][j])
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                rows[r][c] = val
                new_diffs = set()
                for j in range(c):
                    new_diffs.add(val - rows[r][j])
                for i in range(r):
                    for j in range(k + 1):
                        new_diffs.add(abs(val - rows[i][j]))
                
                used_diffs.update(new_diffs)
                
                if backtrack(r, c + 1):
                    return True
                
                used_diffs.difference_update(new_diffs)
                rows[r][c] = 0
            
            return False
        
        return backtrack(0, 1)
    
    # Try increasing M from lower bound
    # Give more time budget to M values closer to the expected solution
    # For (4,7), solution is around 134, so M_lb = 4*28 = 112
    
    current_M = M_lb
    max_M_to_try = min(best_scope - 1, M_lb + 50)  # Don't go too far up
    
    remaining_time = deadline - time.time()
    num_attempts = max_M_to_try - current_M + 1
    if num_attempts <= 0:
        return best_rows
    
    # Allocate time budget: more time for higher M values (closer to solution)
    for M in range(current_M, max_M_to_try + 1):
        if time.time() > deadline - 3:
            break
        
        # Give more time to M values closer to where we expect the solution
        # Assume solution is around M_lb + 20 to M_lb + 40 for (4,7)
        distance_from_expected = abs(M - (M_lb + 30))
        time_budget = time.time() + max(2.0, min(8.0, 10.0 - distance_from_expected * 0.1))
        
        sol = solve_with_randomized_backtracking(M, time_budget)
        
        if sol:
            # Verify solution
            used = set()
            valid = True
            for row in sol:
                if row[0] != 0 or not all(row[i] < row[i+1] for i in range(len(row)-1)):
                    valid = False
                    break
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in used:
                            valid = False
                            break
                        used.add(d)
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                scope = max(max(r) for r in sol)
                if scope < best_scope:
                    best_rows = sol
                    best_scope = scope
                # Since we're iterating M upwards, the first valid solution
                # for a given M is the best for that M
                # But we might find a better one with a different M later?
                # No, M is the upper bound on scope, so smaller M is better.
                # If we found a valid solution for M, it has scope <= M.
                # Since we're going from low M to high M, the first valid one
                # has the smallest M, hence smallest possible scope.
                return best_rows
    
    return best_rows
