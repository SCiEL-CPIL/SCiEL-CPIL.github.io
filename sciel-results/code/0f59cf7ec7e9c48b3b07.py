import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Lower bound on M
    # Total diffs = n_rows * k * (k+1) / 2
    # M must be at least total_diffs
    total_diffs = n_rows * k * (k + 1) // 2
    min_M = total_diffs
    
    # We know best is 112. Let's start from 105.
    # If n_rows=7, k=5 (wait, task says 6 increasing integers, so k=5? 
    # "list of k + 1 integers". 
    # Task: "Choose 7 rows of 6 increasing integers". 
    # So n_rows=7, number of elements=6. 
    # The function signature is construct(n_rows, k). 
    # "Each row is a list of k + 1 integers". 
    # So if we want 6 integers, k=5.
    # The prompt says "evaluator passes this task's n_rows and k".
    # I should assume the evaluator passes n_rows=7 and k=5.
    
    best_solution = None
    
    # Try M from min_M upwards
    # To be safe, cap the search if it takes too long.
    max_M_to_try = min_M + 50 # 105 to 155
    
    for M in range(min_M, max_M_to_try + 1):
        if time.time() > deadline:
            break
            
        # Run a randomized backtracking search for this M
        solution = solve_with_backtracking(n_rows, k, M, deadline)
        if solution:
            return solution
            
    # If we didn't find a solution in the range, 
    # fall back to a greedy construction that might be valid but larger
    # Or return the best found so far (which is None)
    # The greedy from the seed program is a safe fallback to ensure validity
    return greedy_construct(n_rows, k)

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

def solve_with_backtracking(n_rows, k, M, deadline):
    # We will build rows one by one.
    # State: 
    #   rows: list of completed rows
    #   current_row: list of elements in the row being built
    #   used_diffs: set of all differences used so far (from completed rows and current row)
    
    # To improve performance, we can pre-compute valid next steps? 
    # No, dynamic.
    
    # Heuristic: 
    # When building a row, we prefer smaller elements to keep the row compact.
    # But we need to ensure we can finish the row.
    
    # Randomization: 
    # We will perform multiple restarts with different random seeds or strategies.
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            return None
        
        # Reset state
        used_diffs = set()
        rows = []
        current_row = [0]
        # We use a stack for backtracking
        # Stack items: (row_index, current_row_state, used_diffs_state, next_x_to_try)
        # This is memory intensive. 
        # Alternative: Recursive DFS with pruning.
        
        # Let's use a recursive function with a node limit per restart to avoid deep hangs.
        node_limit = 50000
        nodes_visited = 0
        
        def dfs(row_idx, curr_row, curr_used):
            nonlocal nodes_visited
            if time.time() > deadline or nodes_visited > node_limit:
                return None
            
            nodes_visited += 1
            
            if len(curr_row) == k + 1:
                # Row complete
                new_rows = rows + [curr_row]
                new_used = curr_used
                if row_idx + 1 == n_rows:
                    return new_rows
                else:
                    # Start next row
                    return dfs(row_idx + 1, [0], new_used)
            
            # Determine next element
            last_val = curr_row[-1]
            # Candidates: last_val + 1 to M
            # We need to pick a valid x
            # To add randomness, we can shuffle the candidates or pick randomly from valid ones
            # But "smallest first" is a good heuristic for compactness.
            # Let's try a mix: 80% smallest valid, 20% random valid.
            
            valid_candidates = []
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in curr_row}
                # Check conflicts with current row diffs and global used
                if not (new_diffs & curr_used) and not (new_diffs & {b - a for a, b in zip(curr_row, curr_row[1:])}): 
                    # Wait, curr_used should already contain diffs of curr_row?
                    # Let's maintain curr_used as all diffs from all rows so far, including current partial row.
                    pass
            
            # Let's refine the state.
            # curr_used contains diffs from all completed rows AND the current partial row.
            # When adding x, new_diffs = {x - a for a in curr_row}.
            # We must ensure new_diffs are not in curr_used.
            
            candidates = []
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in curr_row}
                if new_diffs & curr_used:
                    continue
                # Also, we must ensure that we can still fit (k+1 - len(curr_row)) more elements.
                # This is always possible if M is large enough, but differences might block it.
                candidates.append(x)
                
            if not candidates:
                return None # Dead end
            
            # Heuristic ordering
            # Sort candidates by value (smallest first)
            # But to allow backtracking to explore other options, we don't just pick the first.
            # We recurse on the first candidate. If it fails, we try the next.
            # This is standard DFS.
            
            # To improve, we can randomize the order of candidates slightly.
            # Or, we can just rely on the restarts to change the path.
            # Since DFS is deterministic for a given state, restarts need to change the state or the choice.
            # Let's shuffle the candidates list slightly? No, that breaks the "smallest first" heuristic which is good for keeping M small.
            # Instead, we can use a "weighted" choice or just accept that DFS might be slow and rely on the node limit.
            
            # Optimization: If we find a solution, we return it.
            # The DFS will find the lexicographically smallest solution (in terms of row elements) for a given M.
            # This is a good property.
            
            # Try candidates in order
            for x in candidates:
                new_row = curr_row + [x]
                new_diffs = {x - a for a in curr_row}
                new_used = curr_used | new_diffs
                res = dfs(row_idx, new_row, new_used)
                if res:
                    return res
                    
            return None

        # Start DFS
        res = dfs(0, [0], set())
        if res:
            return res
            
    return None
