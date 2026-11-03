"""Difference triangle set (7,5) solver using backtracking with time budget."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within rows are distinct across all rows.
    """
    deadline = time.time() + 120  # 120s wall-clock limit with safety margin
    
    # Total differences needed: n_rows * k * (k+1) / 2
    # For n_rows=7, k=5: 7 * 15 = 105 differences
    # Simple counting bound: sqrt(2*105) ≈ 14.49, but that's for a single row
    # For multiple rows, the scope m must satisfy: m*(m-1)/2 >= n_rows * k*(k+1)/2
    # So m >= sqrt(n_rows * k * (k+1))
    
    # Let's use a backtracking approach with increasing scope
    # Start from a reasonable lower bound
    
    # Lower bound on scope: we need n_rows * k*(k+1)/2 distinct differences
    # The maximum difference in any row is at most m (the scope)
    # So we need at least n_rows * k*(k+1)/2 distinct values in [1, m]
    # Thus m >= n_rows * k*(k+1)/2
    # But actually, each row can have at most m*(m-1)/2 distinct differences
    # So we need n_rows * k*(k+1)/2 <= m*(m-1)/2
    # m >= sqrt(n_rows * k*(k+1))
    
    # For n_rows=7, k=5: m >= sqrt(7*15) = sqrt(105) ≈ 10.25
    # But the known best is 112, so the counting bound is much tighter in practice
    # because differences are not just bounded by m, they're specific values
    
    # Actually, the counting bound mentioned in the task is 105, which is
    # n_rows * k*(k+1)/2 = 7 * 15 = 105
    # This suggests that the scope must be at least 105? No, that doesn't make sense.
    
    # Let me reconsider. The "simple counting bound is 105" likely means that
    # the largest entry must be at least 105. But the best known is 112.
    
    # Let's try a backtracking search with increasing scope.
    # We'll start from the counting bound and increase.
    
    # First, let's compute a reasonable starting scope
    # The counting bound: m such that m*(m-1)/2 >= n_rows * k*(k+1)/2
    # m >= sqrt(n_rows * k*(k+1))
    
    # For the actual problem, let's just try scopes from a reasonable minimum
    # up to some maximum, using backtracking.
    
    # Actually, let's use a smarter approach: start with the known best (112)
    # and try to find a better solution, or just use a greedy + local search.
    
    # Given the time constraint, let's use a hybrid approach:
    # 1. Generate initial solution with greedy
    # 2. Try to improve with local search / simulated annealing
    
    # But first, let's try backtracking for small scopes
    
    # For n_rows=7, k=5, the scope is around 112, which is too large for
    # brute-force backtracking. So we need a different approach.
    
    # Let's use a greedy approach with backtracking for row construction,
    # trying to minimize the scope.
    
    def greedy_row(used_diffs, k, max_scope):
        """Construct a single row greedily with backtracking."""
        row = [0]
        
        def backtrack(pos, current_row):
            if time.time() > deadline:
                return None
            if len(current_row) == k + 1:
                return current_row[:]
            
            # Try values from 1 to max_scope
            for x in range(1, max_scope + 1):
                if x <= current_row[-1]:
                    continue
                # Check if all new differences are unused
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                
                # Recurse
                used_diffs.update(new_diffs)
                result = backtrack(pos + 1, current_row + [x])
                if result is not None:
                    return result
                used_diffs.difference_update(new_diffs)
            
            return None
        
        return backtrack(1, row)
    
    # Try increasing scopes
    # Start from a reasonable lower bound
    # The counting bound suggests m >= 105 for this problem
    # Let's start from 105 and go up
    
    # But backtracking with scope 105+ is too slow.
    # Let's use a different strategy: construct rows one by one with greedy,
    # and use the minimum possible scope for each row.
    
    # Actually, let's just use the greedy approach from the seed, but with
    # some improvements: try different orderings or use backtracking for
    # the last few rows.
    
    # For now, let's implement a clean greedy with backtracking for each row,
    # trying to keep the scope as small as possible.
    
    best_solution = None
    best_scope = float('inf')
    
    # Try to construct all rows with a given scope
    # Use backtracking for the entire construction
    
    def try_scope(scope):
        """Try to construct n_rows rows with maximum entry <= scope."""
        used_diffs = set()
        rows = []
        
        def backtrack_rows(row_idx):
            if time.time() > deadline:
                return False
            if row_idx == n_rows:
                rows.append(rows[-1])  # dummy
                return True
            
            # Construct row row_idx
            row = [0]
            
            def backtrack_row(pos, current_row):
                if time.time() > deadline:
                    return None
                if len(current_row) == k + 1:
                    return current_row[:]
                
                # Try values from 1 to scope
                for x in range(1, scope + 1):
                    if x <= current_row[-1]:
                        continue
                    new_diffs = set()
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    
                    used_diffs.update(new_diffs)
                    result = backtrack_row(pos + 1, current_row + [x])
                    if result is not None:
                        return result
                    used_diffs.difference_update(new_diffs)
                
                return None
            
            result = backtrack_row(1, row)
            if result is not None:
                rows.append(result)
                if backtrack_rows(row_idx + 1):
                    return True
                rows.pop()
                # Undo differences
                for a in result:
                    for b in result:
                        if a < b:
                            used_diffs.discard(b - a)
            return False
        
        return backtrack_rows(0)
    
    # This is too slow for scope ~112. Let's use a different approach.
    # Greedy with local search.
    
    def greedy_construct():
        """Greedy construction with backtracking for each row."""
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            # Try to build the row with minimal scope
            # Use backtracking
            def build_row(current_row):
                if time.time() > deadline:
                    return None
                if len(current_row) == k + 1:
                    return current_row[:]
                
                # Try smallest possible next value
                last = current_row[-1]
                for x in range(last + 1, 200):  # Cap at 200 for safety
                    new_diffs = set()
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    
                    used_diffs.update(new_diffs)
                    result = build_row(current_row + [x])
                    if result is not None:
                        return result
                    used_diffs.difference_update(new_diffs)
                
                return None
            
            result = build_row(row)
            if result is not None:
                rows.append(result)
            else:
                return None
        
        return rows
    
    # Try greedy construction
    solution = greedy_construct()
    if solution is not None:
        scope = max(max(row) for row in solution)
        return solution
    
    # Fallback: simple greedy from seed
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
