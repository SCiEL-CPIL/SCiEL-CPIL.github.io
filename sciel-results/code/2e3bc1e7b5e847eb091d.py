import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    row_len = k + 1
    diffs_per_row = row_len * (row_len - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on max mark M: M >= total_diffs_needed
    min_possible_max = total_diffs_needed
    
    def check_time():
        return time.time() > deadline

    def greedy_construct(randomize=False):
        """Constructs a set of rows using greedy approach. If randomize, picks random valid mark."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                found = False
                # Determine candidate range
                # In deterministic greedy, we just increment x.
                # In random greedy, we might look ahead or pick randomly from a window.
                # For simplicity and speed, we stick to incrementing but maybe skip some?
                # Actually, standard greedy is: try x+1, x+2...
                # Randomized: try x+1, x+2... but if multiple work? Only the first works in standard.
                # To randomize, we can pick a random x from [current_last+1, current_last+limit] and check.
                # But finding a valid x is the hard part.
                # Let's stick to deterministic for the base, and use backtracking for improvement.
                # Or, we can randomize the *order* of rows? No, rows are independent sets of diffs.
                
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    found = True
            rows.append(row)
        return rows

    # 1. Get initial solution
    initial_rows = greedy_construct()
    current_best_rows = initial_rows
    current_best_max = max(max(r) for r in initial_rows)
    
    # 2. Try to improve with backtracking
    # We try to find a solution with max mark < current_best_max
    
    def backtrack_search(target_max):
        """
        Tries to construct n_rows rows with max mark <= target_max.
        Returns a list of rows if found, else None.
        """
        used = set()
        rows = []
        
        # Pre-check: Is it even possible?
        # Available diffs in [1, target_max] is target_max.
        # We need total_diffs_needed.
        if target_max < total_diffs_needed:
            return None
            
        def build_row(row_idx):
            if row_idx == n_rows:
                return True
            
            row = [0]
            # We need to fill row with row_len marks.
            # The last mark must be <= target_max.
            
            # Heuristic: Try to place the next mark.
            # To make it faster, we can fix the last mark of the row first?
            # Or just build sequentially.
            
            # Let's use a recursive helper for the current row
            def build_current_row(current_row, used_local):
                if len(current_row) == row_len:
                    # Check if max is within target
                    if current_row[-1] > target_max:
                        return False
                    rows.append(current_row)
                    if build_row(row_idx + 1):
                        return True
                    rows.pop()
                    return False
                
                last_mark = current_row[-1]
                # Start from last_mark + 1
                # Upper bound for next mark: target_max
                # But we also need to ensure we can fit the remaining marks.
                # Remaining marks needed: row_len - len(current_row)
                # Min space needed: (row_len - len(current_row)) - 1 (if we pack tight)
                # So next_mark <= target_max - (remaining - 1) ? Not strictly, but a good bound.
                
                start_val = last_mark + 1
                end_val = target_max
                
                # Optimization: If we have many marks left, we can't start too late.
                # Max possible last mark is target_max.
                # If we are at index i (0-indexed, 0 is first), we have row_len - 1 - i marks left to place.
                # The smallest possible last mark if we pack tight is last_mark + (row_len - 1 - i).
                # So we must have last_mark + (row_len - 1 - i) <= target_max.
                # This is implicitly handled if we just iterate up to target_max.
                
                # To avoid deep recursion issues, we can limit the search depth or branching.
                # Let's iterate candidates.
                # Randomize candidate order to escape local minima?
                candidates = list(range(start_val, end_val + 1))
                if random.random() < 0.5:
                    random.shuffle(candidates)
                else:
                    # Try smaller ones first (compact)
                    pass 
                
                for x in candidates:
                    if check_time():
                        return False
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_local:
                        continue
                    
                    # Pruning: Check if we have enough "space" left?
                    # Hard to estimate.
                    
                    current_row.append(x)
                    used_local.update(new_diffs)
                    
                    if build_current_row(current_row, used_local):
                        return True
                    
                    current_row.pop()
                    used_local.difference_update(new_diffs)
                
                return False
            
            # Start building this row
            # We can try different starting strategies for the row?
            # For now, just build from 0.
            if build_current_row(row, used):
                return True
            
            return False
        
        return build_row(0)

    # Try to find a better solution
    # We iterate downwards from current_best_max - 1
    # To prevent one search from eating all time, we limit each search to a fraction of remaining time
    # or just let it run until deadline and hope it finds one or fails fast.
    # Better: Limit each search to ~5-10 seconds, then restart with different random seeds if it fails.
    
    search_time_limit = 5.0
    max_searches = 20 # Arbitrary cap to avoid infinite loops if search is very fast but failing
    
    search_count = 0
    target = current_best_max - 1
    
    while target >= min_possible_max and not check_time():
        search_start = time.time()
        # Run backtracking with a time limit
        # We modify check_time to respect a local deadline
        local_deadline = search_start + search_time_limit
        
        # We need to pass local_deadline to the inner functions or use a global flag
        # Let's use a simple flag
        search_aborted = [False]
        
        def local_check_time():
            if time.time() > local_deadline:
                search_aborted[0] = True
                return True
            return False
        
        # Monkey patch or pass? Passing is cleaner but we defined check_time inside construct.
        # Let's just rely on the global deadline for now, and add a local break condition in the loops if possible.
        # Actually, the `check_time` inside `backtrack_search` uses the global `deadline`.
        # We should make it respect a local limit.
        
        # Let's refactor to pass a deadline.
        
        found = False
        # We'll just run the search. If it takes too long, the global check_time will trigger and return False.
        # But we want to try multiple targets.
        # So we need to abort the search if it takes > search_time_limit.
        
        # Simple approach: Run search. If it returns None and time spent < limit, try next target?
        # No, if it returns None, it might be because it was exhaustive (failed) or aborted.
        # If it was aborted, we should try again with randomization or a different target?
        # If it was exhaustive, we know no solution exists for this target (assuming correct logic).
        
        # Let's assume the search is fast enough for small n, k.
        # If n, k are large, backtracking will blow up.
        # The greedy is O(n * k * M). M is around 100-200.
        # Backtracking is exponential.
        
        # So we should only use backtracking if the problem size is small enough.
        # Estimate: n_rows * row_len. If < 20, backtracking is viable.
        
        if n_rows * row_len <= 25:
            # Try to find solution for 'target'
            # We need to handle the time limit properly.
            # Let's just call it. If it hits the global deadline, it will stop.
            # To allow multiple attempts, we can reduce the global deadline temporarily? No.
            # Let's just try one target at a time.
            
            # To make it robust, we'll just try the immediate next lower target.
            # If it finds one, we update best and continue.
            # If it doesn't, we decrement target and try again.
            
            # To prevent one search from taking 100s, we can't easily interrupt the pure Python recursion without checks.
            # The check_time() calls are in the innermost loop.
            
            res = backtrack_search(target)
            if res:
                current_best_rows = res
                current_best_max = max(max(r) for r in res)
                # We found a better one. We can try to go even lower.
                target = current_best_max - 1
                search_count = 0 # Reset counter
            else:
                # No solution found for this target (either exhausted or timeout)
                # If it was a timeout, we might want to try a slightly higher target?
                # No, we are iterating down. If we timed out, we can't trust that no solution exists.
                # But we also can't spend all time on one target.
                # Let's just break if we've been searching for a while without success.
                target -= 1
                search_count += 1
                if search_count > max_searches:
                    break
        else:
            # Problem too large for backtracking. Rely on greedy/local search.
            # We can try to improve the greedy solution with local search.
            # For now, just return the greedy one.
            break

    return current_best_rows
