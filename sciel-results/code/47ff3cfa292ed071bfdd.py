import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # 1. Get an initial solution using the greedy method to establish an upper bound (best_scope)
    # This ensures we have a valid solution to return even if the search fails or times out.
    
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    initial_rows = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in initial_rows)
    best_solution = initial_rows
    
    # 2. Backtracking search to find a solution with a smaller scope
    # We will try to find a solution where the max entry is < best_scope.
    # We use a global deadline.
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds to leave margin for overhead
    
    # To make the search effective, we need to order the search space.
    # We construct rows one by one.
    # We use a recursive DFS.
    
    # Optimization: Pre-compute or cache? 
    # The state space is small enough for n_rows=4, k=7.
    # Max scope is around 134. 
    # Branching factor: At each step, we try x. 
    # Pruning is crucial.
    
    used_diffs = set()
    rows = []
    
    # We will modify best_solution in place if we find a better one.
    # We need to be careful with recursion depth and state management.
    
    def dfs(row_idx, pos, current_row):
        nonlocal best_scope, best_solution, used_diffs, rows
        
        # Time check
        if time.time() > deadline:
            return False
        
        # If we have constructed all rows
        if row_idx == n_rows:
            # Calculate scope
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_solution = [r[:] for r in rows]
            return True
        
        # If we are starting a new row
        if pos == 0:
            current_row = [0]
            return dfs(row_idx, 1, current_row)
        
        # If we have completed the current row
        if pos == k + 1:
            rows.append(current_row)
            success = dfs(row_idx + 1, 0, [])
            rows.pop()
            return success
        
        # We are at position 'pos' in row 'row_idx'
        # The last element is current_row[-1]
        last_val = current_row[-1]
        
        # Lower bound check: 
        # We need (k + 1 - pos) more elements.
        # The smallest possible values are last_val+1, last_val+2, ...
        # So the final value will be at least last_val + (k + 1 - pos)
        # If this is >= best_scope, we can prune.
        min_possible_final = last_val + (k + 1 - pos)
        if min_possible_final >= best_scope:
            return False
        
        # Try candidates x
        # Start from last_val + 1
        # We can also start from 1 if pos==1, but last_val is 0 so last_val+1 is 1.
        
        # Optimization: We can limit the range of x.
        # x must be < best_scope.
        # Also, the differences x - y must not be in used_diffs.
        
        for x in range(last_val + 1, best_scope):
            # Check differences
            valid = True
            new_diffs = []
            for y in current_row:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Additional pruning: 
            # If we add x, the new diffs are new_diffs.
            # Are there any immediate conflicts with future requirements? 
            # Hard to say without lookahead.
            
            # Commit
            used_diffs.update(new_diffs)
            current_row.append(x)
            
            # Recurse
            success = dfs(row_idx, pos + 1, current_row)
            
            # Undo
            current_row.pop()
            for d in new_diffs:
                used_diffs.remove(d)
            
            if success:
                return True
                
        return False

    # Start the search
    # We start with an empty used_diffs and empty rows
    # But we need to initialize the first row.
    # The dfs function handles pos=0 by setting current_row=[0].
    
    # Note: The greedy solution might be optimal for small cases, but for (4,7) it is not.
    # The greedy solution for (4,7) typically has a scope around 150-160.
    # The optimal is 134.
    
    # To make the search faster, we can sort the rows or use other heuristics?
    # Since the rows are symmetric in the constraint (differences are distinct across ALL rows),
    # the order of rows doesn't matter for validity, but it matters for the search path.
    # However, since we are minimizing the global max, and the rows are constructed sequentially,
    # the "first" row in our search order might end up with larger numbers if we are not careful?
    # No, we are trying to keep numbers small.
    
    # One issue: The DFS above explores the first row first.
    # If the first row takes up many small differences, it might force the second row to use larger numbers.
    # Is there a better ordering?
    # Maybe we should construct the rows in a way that balances the usage of differences?
    # For now, simple DFS with smallest-first is a good structural change from pure greedy.
    
    # Let's run the DFS
    # We need to reset state for the DFS
    used_diffs = set()
    rows = []
    
    # We call dfs for the first row
    # dfs(0, 0, []) will set current_row to [0] and call dfs(0, 1, [0])
    dfs(0, 0, [])
    
    # Return the best solution found
    return best_solution

# EVOLVE-BLOCK-END
