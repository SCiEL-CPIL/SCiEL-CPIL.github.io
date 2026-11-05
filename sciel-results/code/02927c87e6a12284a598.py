import sys
import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    """
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Best known for (7,5) is 112.
    # We aim to find a solution with scope <= 112, ideally lower.
    
    def is_valid_solution(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            # Check increasing
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return False
            # Check diffs
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def find_solution(max_scope, seed=0):
        """
        Try to find a valid set of rows with max entry <= max_scope.
        Uses backtracking with randomization.
        """
        random.seed(seed)
        used = set()
        rows = []
        
        # Order of rows to build. We can shuffle this to break symmetry.
        row_indices = list(range(n_rows))
        random.shuffle(row_indices)
        
        def backtrack(idx):
            if idx == n_rows:
                return True
            
            r = row_indices[idx]
            
            # Try to find a row for position r
            # We want to minimize the max element, so we try smaller spans first?
            # Or we can just find *any* valid row and then optimize?
            # Given the global constraint, finding *any* valid row with span <= max_scope is the goal.
            
            # To be efficient, we can generate candidate rows.
            # Since k=5, the row is [0, a1, a2, a3, a4, a5] with a5 <= max_scope.
            
            # We'll use a recursive search for the row elements.
            def find_row():
                row = [0]
                
                def build_row(pos, current_row):
                    if pos == k:
                        # Check if all diffs are unused
                        # Note: we check diffs incrementally in the loop below, 
                        # but we need to ensure the *last* step didn't conflict.
                        # Actually, we check conflicts when adding a new element.
                        # So if we reach pos==k, the row is valid w.r.t used diffs.
                        return current_row[:]
                    
                    last = current_row[-1]
                    # Remaining elements to add: k - pos
                    # Max value for next element: max_scope
                    # Min value: last + 1
                    
                    # Heuristic: Try values in a random order or sorted order?
                    # Sorted order (ascending) tends to find small spans quickly.
                    # But it might get stuck.
                    # Let's try ascending, but with a random shuffle of the *set* of candidates?
                    # No, the candidates depend on 'last'.
                    
                    # Candidates: x such that last < x <= max_scope
                    # And x - a_i not in used for all a_i in current_row
                    # And x - a_i not in {current_row[j] - current_row[i] for i<j<=pos} (local diffs)
                    
                    # Optimization: Pre-check if there are enough unused diffs?
                    # Hard to do quickly.
                    
                    candidates = []
                    for x in range(last + 1, max_scope + 1):
                        # Check local diffs
                        valid_local = True
                        local_diffs = set()
                        for a in current_row:
                            d = x - a
                            if d in local_diffs: # Should not happen if x > a
                                valid_local = False
                                break
                            local_diffs.add(d)
                            if d in used:
                                valid_local = False
                                break
                        if valid_local:
                            candidates.append(x)
                    
                    if not candidates:
                        return None
                    
                    # Shuffle candidates to avoid deterministic traps
                    random.shuffle(candidates)
                    
                    for x in candidates:
                        current_row.append(x)
                        # Update used? No, we need to backtrack.
                        # We can pass a new set or modify and restore.
                        # Modify and restore is faster.
                        
                        new_diffs = set()
                        for a in current_row[:-1]: # Exclude x itself? No, x is last.
                            # Wait, current_row now includes x.
                            # Diffs are x - a for a in current_row[:-1]
                            pass
                            
                        # Let's re-calculate new diffs added by x
                        added_diffs = set()
                        for a in current_row[:-1]:
                            added_diffs.add(x - a)
                        
                        # Check if any added_diff are in used
                        # We already checked this in the candidate loop?
                        # Yes, "if d in used" was checked.
                        
                        # So we can just add them to used
                        used.update(added_diffs)
                        
                        result = build_row(pos + 1, current_row)
                        if result is not None:
                            return result
                            
                        # Backtrack
                        used.difference_update(added_diffs)
                        current_row.pop()
                        
                    return None
                
                return build_row(1, [0])
            
            # Try to find a row
            # We might need to try multiple times or different strategies if the first one fails?
            # The backtracking inside find_row is exhaustive for the given max_scope.
            # If it returns None, no row exists with max_scope <= max_scope given the current used set.
            
            row = find_row()
            if row is None:
                return False
            
            rows.append(row)
            
            if backtrack(idx + 1):
                return True
            
            # Backtrack
            # Remove diffs of the row we just added
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.discard(row[j] - row[i])
            rows.pop()
            
            return False
        
        return backtrack(0)

    # Strategy:
    # 1. Try to find a solution with scope = counting_bound (105).
    # 2. If fail, try 106, 107, ... up to 120.
    # 3. Use multiple seeds for each scope to increase chance of finding a solution.
    
    # Counting bound for (7,5) is 105.
    # Let's start from 105.
    
    best_rows = None
    best_scope = float('inf')
    
    # We know 112 is possible.
    # Let's try 105 to 112.
    
    for scope in range(105, 113):
        # Try a few seeds
        for seed in range(10):
            if find_solution(scope, seed):
                # Verify
                if is_valid_solution(rows):
                    actual_max = max(row[-1] for row in rows)
                    if actual_max < best_scope:
                        best_scope = actual_max
                        best_rows = [row[:] for row in rows]
                    # If we found a solution with this scope, we can break?
                    # No, we want the *minimum* scope.
                    # Since we iterate scope increasing, the first valid scope is the best?
                    # Not necessarily, because find_solution(scope) might fail even if a solution exists
                    # due to the specific search path (though backtracking should be complete).
                    # If backtracking is complete, it will find a solution if one exists.
                    # So if find_solution(scope) returns True, a solution exists.
                    # We can break the seed loop and the scope loop.
                    break
        if best_rows is not None and best_scope == scope:
            break
            
    if best_rows is not None:
        return best_rows
    
    # Fallback: Return the known best or a greedy result
    # If we didn't find anything, return a simple greedy construction
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
