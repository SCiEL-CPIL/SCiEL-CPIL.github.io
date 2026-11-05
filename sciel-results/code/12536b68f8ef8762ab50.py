# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Strategy:
    1. Greedy baseline for a valid initial solution.
    2. Local Search (Hill Climbing) to reduce the maximum entry.
       - State: list of rows.
       - Objective: minimize max(max(r) for r in rows).
       - Move: Rebuild one row greedily, trying to keep it as compact as possible
         given the differences already used by other rows.
       - Acceptance: Accept if the new max is strictly smaller, or occasionally
         accept equal/smaller to escape local minima (simulated annealing-like).
    3. Return the best solution found within the time limit.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # --- Greedy Baseline ---
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
    
    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # --- Local Search / Hill Climbing ---
    
    # Pre-compute all pairs (i, j) for i < j within a row of size k+1
    # This helps in efficiently checking differences
    # Actually, we just iterate.
    
    def get_used_diffs(rows, exclude_row_idx=None):
        """
        Returns a set of all differences used in rows, excluding the specified row.
        """
        used = set()
        for idx, row in enumerate(rows):
            if idx == exclude_row_idx:
                continue
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def try_rebuild_row(row_idx, rows, used_others, max_limit, rng):
        """
        Attempts to rebuild row at row_idx such that its max entry is <= max_limit.
        Returns the new row if successful, else None.
        Uses a greedy approach with some randomization to escape local minima.
        """
        # We want to build a row starting with 0, increasing, max <= max_limit.
        # We also must not conflict with 'used_others'.
        
        # To encourage compactness, we try to pick the smallest valid x first,
        # but sometimes pick a larger one to avoid blocking future steps?
        # Actually, for a single row rebuild, we just need *any* valid row within the limit.
        # To minimize the *global* max, we ideally want this row to be as small as possible.
        
        # Heuristic: 
        # 1. Try to build the row greedily from left to right.
        # 2. At each step, candidates are x > last, x <= max_limit - (remaining - 1).
        # 3. Filter candidates that don't conflict with used_others or current row diffs.
        # 4. Pick the smallest candidate? Or random?
        #    Picking the smallest makes the row compact, which is good for the global max.
        #    However, it might lead to a dead end later in the row.
        #    So we use backtracking.
        
        def dfs_rebuild(current_row, current_used_local, remaining_slots):
            if remaining_slots == 0:
                return True
            
            last_val = current_row[-1]
            min_x = last_val + 1
            max_x = max_limit - (remaining_slots - 1)
            
            if min_x > max_x:
                return False
            
            # Generate candidates
            # Optimization: Instead of iterating all x from min_x to max_x,
            # we can be smarter, but for small k (k=7, len=8), this is fine.
            
            # We want to try smaller x first to keep the row compact.
            # But if we always pick the smallest, we might get stuck.
            # Let's iterate and try each one with backtracking.
            
            # To add diversity, we can shuffle the candidates if we have many?
            # Or just iterate. Since we are doing backtracking, order matters for finding *a* solution,
            # but we want the *smallest* solution. 
            # If we find a solution, is it the smallest? 
            # If we iterate x from min_x upwards, and use DFS, the first solution found
            # will have the smallest possible first element, then smallest possible second, etc.
            # This yields a lexicographically smallest row, which generally minimizes the max.
            
            for x in range(min_x, max_x + 1):
                # Check conflicts with global used
                # Check conflicts with local row
                conflict = False
                for a in current_row:
                    d = x - a
                    if d in used_others or d in current_used_local:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Try this x
                new_local_used = current_used_local | {x - a for a in current_row}
                current_row.append(x)
                
                if dfs_rebuild(current_row, new_local_used, remaining_slots - 1):
                    return True
                
                # Backtrack
                current_row.pop()
            
            return False
        
        new_row = [0]
        # local used is empty initially
        if dfs_rebuild(new_row, set(), k):
            return new_row
        return None

    def local_search(rows, max_limit, time_budget, rng):
        """
        Performs local search to find a configuration with max entry <= max_limit.
        Returns True if found, and updates 'rows' in place.
        """
        deadline = time.time() + time_budget
        current_rows = [r[:] for r in rows]
        
        # We iterate: pick a row, try to rebuild it with a tighter or same limit.
        # To improve the global max, we need at least one row to have a smaller max.
        
        # Strategy:
        # 1. Calculate current max of each row.
        # 2. Identify rows that are "large" (close to global max).
        # 3. Try to rebuild them with a limit slightly smaller than the current global max.
        # 4. If successful, the global max might decrease.
        
        # We will do a loop of attempts.
        attempts = 0
        
        while time.time() < deadline:
            attempts += 1
            if attempts > 10000: # Safety break
                break
                
            # Identify the row with the largest max entry
            max_vals = [max(r) for r in current_rows]
            global_max = max(max_vals)
            
            # Pick a row that has the global max (or close to it) to try to shrink.
            # Randomly pick among the largest ones to add diversity.
            candidates_idx = [i for i, m in enumerate(max_vals) if m >= global_max - 2]
            if not candidates_idx:
                candidates_idx = [max(range(len(max_vals)), key=lambda i: max_vals[i])]
            
            row_idx = rng.choice(candidates_idx)
            
            # We want to shrink this row.
            # Let's try to rebuild it with a limit = global_max - 1.
            # If that fails, try global_max - 2, etc?
            # Or just try to rebuild with the current global_max first to ensure validity,
            # then try to tighten?
            
            # Better: Try to rebuild with limit = global_max - 1.
            # If it succeeds, the new global max is at most global_max - 1 (assuming other rows don't exceed it).
            # Wait, if other rows have max = global_max, and we shrink one, the global max stays global_max.
            # We need to shrink *all* rows that are at the global max to reduce the global max.
            
            # Revised Strategy:
            # 1. Let M = current global max.
            # 2. Identify all rows with max == M.
            # 3. Try to rebuild *one* of them with limit M-1.
            # 4. If successful, the set of rows with max == M shrinks by 1.
            # 5. Repeat until no rows have max == M. Then M becomes M-1.
            
            # This is a sequential improvement.
            
            # Find rows with max == global_max
            rows_at_max = [i for i, m in enumerate(max_vals) if m == global_max]
            
            if not rows_at_max:
                # Should not happen if global_max is max(max_vals)
                break
            
            # Pick one to shrink
            target_row_idx = rng.choice(rows_at_max)
            
            # Calculate used diffs excluding this row
            used_others = get_used_diffs(current_rows, exclude_row_idx=target_row_idx)
            
            # Try to rebuild with limit = global_max - 1
            new_row = try_rebuild_row(target_row_idx, current_rows, used_others, global_max - 1, rng)
            
            if new_row is not None:
                # Check if this actually reduces the max of this row
                if max(new_row) < global_max:
                    current_rows[target_row_idx] = new_row
                    # We made progress! The global max might drop if this was the last row at global_max.
                    # We don't need to do anything else, the loop will re-evaluate.
                else:
                    # It didn't shrink below global_max, but it's a valid row.
                    # Maybe we can accept it to change the structure?
                    # If we accept it, we might escape a local minimum where this row was "stuck".
                    # Let's accept it with some probability, or if it's different.
                    if new_row != current_rows[target_row_idx]:
                         current_rows[target_row_idx] = new_row
            
            # If we couldn't shrink, we might try to rebuild with limit = global_max
            # to change the internal structure, which might allow shrinking later.
            # This is "sideways" move.
            else:
                # Try to rebuild with same limit but different structure
                # This helps escape local minima where the greedy rebuild fails.
                new_row = try_rebuild_row(target_row_idx, current_rows, used_others, global_max, rng)
                if new_row is not None and new_row != current_rows[target_row_idx]:
                    # Accept sideways move with some probability (e.g., 0.5) to diversify
                    if rng.random() < 0.5:
                        current_rows[target_row_idx] = new_row
        
        # After the loop, check if we improved the global max
        new_global_max = max(max(r) for r in current_rows)
        if new_global_max < max(max(r) for r in rows):
            rows[:] = current_rows
            return True
        return False

    # Main Local Search Loop
    # We try to reduce the max entry step by step.
    # Start from best_max.
    
    # We can also try random restarts of the whole process if we get stuck?
    # For now, just run local search for the remaining time.
    
    rng = random.Random(42)
    
    # We will run local search until time runs out or we hit the lower bound.
    # Since local search is stochastic, we might want to run it in chunks.
    
    # Initial run
    # To ensure we don't get stuck in a specific local minimum, we can do multiple passes.
    
    while time.time() < start_time + time_limit:
        # Check if we hit the lower bound
        if best_max <= lb:
            break
        
        # Run local search for a small chunk of time
        chunk_time = min(5.0, (start_time + time_limit) - time.time())
        if chunk_time < 0.5:
            break
            
        # We pass the current best_rows to local_search.
        # local_search modifies it in place if it finds an improvement.
        # But our local_search function above takes 'rows' and returns True/False.
        # Let's adapt it to work on a copy and return the new rows if improved.
        
        # Actually, the implementation above modifies 'current_rows' internally and only updates 'rows' if improved.
        # This is safe.
        
        improved = local_search(best_rows, best_max, chunk_time, rng)
        
        if improved:
            best_max = max(max(r) for r in best_rows)
            # We can also try to tighten the bound further in the next iteration.
        else:
            # If no improvement in this chunk, we might be stuck.
            # We can try to perturb the solution or just continue.
            # Continuing is fine as rng changes the choices.
            pass

    # Final Validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        # Fallback to greedy if something went wrong
        best_rows = greedy_construct(n_rows, k)
        best_max = max(max(r) for r in best_rows)
        
    return best_rows

# EVOLVE-BLOCK-END
