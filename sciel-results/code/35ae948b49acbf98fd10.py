"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    
    Strategy: Iterative Deepening DFS (IDDFS) with pruning.
    We try to find a valid configuration with maximum value M, starting from the lower bound.
    """
    start_time = time.time()
    time_limit = 115  # seconds, leave buffer for safety
    
    # Lower bound on scope M
    # Total differences needed: n_rows * k * (k+1) / 2
    # Max differences in [1, M] is M
    # So M >= n_rows * k * (k+1) / 2
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    lb = total_diffs
    
    # We will search for M starting from lb
    # To avoid infinite loops or getting stuck in one bad branch, we add randomness
    # or multiple restarts with different candidate orderings if deterministic fails.
    
    best_rows = None
    best_scope = float('inf')
    
    # Fallback greedy result to ensure we return something valid
    def greedy_fallback():
        used = set()
        rows = []
        for r in range(n_rows):
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

    # First, get a valid baseline using greedy (might have large scope)
    fallback_rows = greedy_fallback()
    best_scope = max(max(row) for row in fallback_rows)
    best_rows = [row[:] for row in fallback_rows]
    
    # Now try to improve using DFS with increasing M
    # We try M from lb to best_scope - 1
    # If we find a solution with M < best_scope, we update best.
    
    # To speed up DFS, we can use a bitset or set for used differences.
    # Since M is around 100-200, the set of differences is small.
    
    def dfs(row_idx, pos_in_row, current_row, used_diffs, M):
        """
        Recursive DFS to construct rows.
        Returns True if a valid completion is found.
        """
        if row_idx == n_rows:
            return True
        
        if pos_in_row == k + 1:
            # Move to next row
            return dfs(row_idx + 1, 0, [0], used_diffs, M)
        
        last_val = current_row[-1]
        # Try candidates from last_val + 1 to M
        # Heuristic: try smaller values first to keep scope small
        for cand in range(last_val + 1, M + 1):
            # Check conflicts
            valid = True
            new_diffs = []
            for a in current_row:
                d = cand - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Pruning: Check if we can still fit the remaining elements?
                # Simple pruning: The smallest possible next value is cand+1.
                # We need to place (k+1 - pos_in_row - 1) more elements.
                # This check is weak, so we rely on backtracking.
                
                # Add new differences
                for d in new_diffs:
                    used_diffs.add(d)
                
                current_row.append(cand)
                
                if dfs(row_idx, pos_in_row + 1, current_row, used_diffs, M):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
        
        return False

    # We iterate M from lb upwards.
    # To handle cases where deterministic DFS is slow or fails due to bad ordering,
    # we can introduce randomization in the candidate order for later attempts.
    # However, for small n_rows and k, deterministic DFS with smallest-first is usually good.
    
    # Optimization: If n_rows * k is large, DFS might be too slow.
    # But for the given task (4, 7), it should be fine.
    
    # We'll try a few random restarts with shuffled candidate orders if the first few M's fail quickly?
    # No, let's just increase M. If DFS for M takes too long, we break.
    
    # To prevent DFS from running too long for a specific M, we can add a timeout check inside DFS?
    # That's hard in pure recursion. Instead, we limit the number of nodes visited?
    # Or we just rely on the fact that if M is close to LB, the tree is pruned heavily.
    
    # Let's add a node limit to DFS to avoid hanging.
    node_limit = 100000
    nodes_visited = 0
    
    def dfs_limited(row_idx, pos_in_row, current_row, used_diffs, M):
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited > node_limit:
            return False
        
        if row_idx == n_rows:
            return True
        
        if pos_in_row == k + 1:
            return dfs_limited(row_idx + 1, 0, [0], used_diffs, M)
        
        last_val = current_row[-1]
        
        # Generate candidates
        candidates = range(last_val + 1, M + 1)
        
        # Optional: Randomize candidate order slightly for diversity in subsequent attempts?
        # For now, keep it deterministic for the first pass.
        
        for cand in candidates:
            valid = True
            new_diffs = []
            for a in current_row:
                d = cand - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                for d in new_diffs:
                    used_diffs.add(d)
                current_row.append(cand)
                
                if dfs_limited(row_idx, pos_in_row + 1, current_row, used_diffs, M):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
        
        return False

    # Try M from lb to best_scope - 1
    for M in range(lb, best_scope):
        if time.time() - start_time > time_limit:
            break
        
        nodes_visited = 0
        used_diffs = set()
        current_row = [0]
        
        # We need to store the solution if found
        # Since dfs_limited modifies current_row, we need to capture it.
        # Let's use a wrapper that stores the solution.
        
        solution_found = [None]
        
        def dfs_capture(row_idx, pos_in_row, current_row, used_diffs, M):
            nonlocal nodes_visited
            nodes_visited += 1
            if nodes_visited > node_limit:
                return False
            
            if row_idx == n_rows:
                # Store a copy of the current state
                # We need to reconstruct the full rows.
                # The current_row is just the last row. We need to track all rows.
                # This is a flaw in the simple DFS state.
                # We need to pass the list of completed rows.
                solution_found[0] = [row[:] for row in current_rows] + [current_row[:]]
                return True
            
            if pos_in_row == k + 1:
                # Move to next row
                return dfs_capture(row_idx + 1, 0, [0], used_diffs, M)
            
            last_val = current_row[-1]
            
            for cand in range(last_val + 1, M + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = cand - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    for d in new_diffs:
                        used_diffs.add(d)
                    current_row.append(cand)
                    
                    if dfs_capture(row_idx, pos_in_row + 1, current_row, used_diffs, M):
                        return True
                    
                    current_row.pop()
                    for d in new_diffs:
                        used_diffs.remove(d)
            
            return False

        # We need to track current_rows
        current_rows = []
        
        # The DFS above doesn't update current_rows correctly when moving to next row.
        # Let's fix the DFS to maintain a list of rows.
        
        def dfs_full(row_idx, pos_in_row, current_row, used_diffs, M):
            nonlocal nodes_visited
            nodes_visited += 1
            if nodes_visited > node_limit:
                return False
            
            if row_idx == n_rows:
                solution_found[0] = [row[:] for row in current_rows] + [current_row[:]]
                return True
            
            if pos_in_row == k + 1:
                # Commit current row
                current_rows.append(current_row[:])
                if dfs_full(row_idx + 1, 0, [0], used_diffs, M):
                    return True
                current_rows.pop()
                return False
            
            last_val = current_row[-1]
            
            for cand in range(last_val + 1, M + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = cand - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    for d in new_diffs:
                        used_diffs.add(d)
                    current_row.append(cand)
                    
                    if dfs_full(row_idx, pos_in_row + 1, current_row, used_diffs, M):
                        return True
                    
                    current_row.pop()
                    for d in new_diffs:
                        used_diffs.remove(d)
            
            return False

        used_diffs = set()
        current_row = [0]
        current_rows = []
        
        if dfs_full(0, 0, current_row, used_diffs, M):
            if solution_found[0]:
                scope = max(max(row) for row in solution_found[0])
                if scope < best_scope:
                    best_scope = scope
                    best_rows = solution_found[0]
                    # We can stop if we hit the lower bound
                    if best_scope == lb:
                        break
                    # We can also break if we found a solution, as we are iterating M upwards
                    # The first solution found for a given M is valid.
                    # Since we iterate M upwards, the first M that yields a solution is the minimum M.
                    break

    return best_rows
# EVOLVE-BLOCK-END
