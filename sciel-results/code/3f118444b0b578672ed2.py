"""Difference triangle set construction via randomized backtracking."""
import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound for M
    # M*(M+1)/2 >= total_diffs
    M_min = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    # Known best for (7,5) is 112. 
    # We will iterate M from M_min upwards.
    # Given the time limit, we might not reach the optimal if M_min is too low and search is hard.
    # But for (7,5), the gap between 105 and 112 is small.
    
    start_time = time.time()
    time_limit = 115  # leave margin for 120s
    
    # Helper to check if a row is a valid Sidon sequence (differences distinct within row)
    # and returns the set of differences
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return None
                diffs.add(d)
        return diffs

    # Backtracking search for a specific M
    def search_rows(M, seed):
        rng = random.Random(seed)
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx):
            if time.time() - start_time > time_limit:
                return False
                
            if row_idx == n_rows:
                return True
            
            # Try to build the row at index row_idx
            # We can randomize the order of rows if we treat them as a set, 
            # but here we fix row order to simplify.
            # To break symmetry, we can try different starting points or random orders for elements.
            
            row = [0]
            
            def build_row(pos, current_row, current_diffs):
                if time.time() - start_time > time_limit:
                    return False
                    
                if pos == k + 1:
                    # Row complete, check against global used_diffs
                    if current_diffs & used_diffs:
                        return False
                    
                    # Commit
                    rows.append(list(current_row))
                    used_diffs.update(current_diffs)
                    
                    if backtrack(row_idx + 1):
                        return True
                        
                    # Undo
                    rows.pop()
                    used_diffs -= current_diffs
                    return False
                
                # Candidates for next element
                # Must be > current_row[-1] and <= M
                start_val = current_row[-1] + 1
                end_val = M
                
                # Optimization: We need to fit (k+1 - pos) more elements.
                # The minimum possible max value for the remaining elements if we pick smallest possible is:
                # current_row[-1] + (k+1 - pos)
                # This is a weak bound. A stronger bound:
                # The differences must fit in the remaining space? No, differences can be large.
                
                # Randomize candidates
                candidates = list(range(start_val, end_val + 1))
                rng.shuffle(candidates)
                
                # Heuristic: Prefer smaller values to keep M low? 
                # Actually, we are fixing M, so we just need to find *any* valid row.
                # Smaller values might leave more room for differences?
                # Let's try smaller values first but with randomization.
                
                for x in candidates:
                    # Check internal validity of current_row + [x]
                    new_diffs = {x - a for a in current_row}
                    
                    # Check for duplicates within new_diffs
                    if len(new_diffs) != len(current_row):
                        continue
                        
                    # Check for duplicates with existing current_diffs
                    if new_diffs & current_diffs:
                        continue
                        
                    # Check for duplicates with global used_diffs (early pruning)
                    # Note: We only need to check against used_diffs when the row is complete?
                    # No, if a difference in the partial row conflicts with a used_diff, 
                    # it will definitely conflict later. So we can prune now.
                    if new_diffs & used_diffs:
                        continue
                        
                    # Recurse
                    current_row.append(x)
                    current_diffs = current_diffs | new_diffs
                    
                    if build_row(pos + 1, current_row, current_diffs):
                        return True
                    
                    current_row.pop()
                    current_diffs = current_diffs - new_diffs # Set difference, creates new set. Inefficient.
                    # Better: pass set by reference and mutate, or copy.
                    # For small sets, copying is fine.
                    # Let's optimize: don't re-create set, just remove the added ones?
                    # Actually, `current_diffs` was passed by value (set is immutable in this context of reassignment? No, sets are mutable).
                    # If I do `current_diffs = current_diffs | new_diffs`, I create a new set.
                    # So `current_diffs` in the caller is unchanged. This is safe but allocates memory.
                    # Given small k (5), this is acceptable.
                    
                return False
            
            return build_row(1, row, set())
        
        # Try multiple random seeds for the same M
        for seed in range(1000):
            if time.time() - start_time > time_limit:
                break
            # Reset state for each attempt?
            # The `rows` and `used_diffs` are local to `search_rows` call?
            # No, they are defined in `search_rows` scope.
            # But `backtrack` modifies them.
            # We need to reset them for each seed attempt.
            
            # Let's refactor: `search_rows` should try one seed.
            # We will loop seeds outside.
            
            # Actually, the current structure defines `rows` and `used_diffs` once.
            # If `backtrack` fails, `rows` and `used_diffs` might be partially modified?
            # The backtracking logic undoes changes (`rows.pop()`, `used_diffs -= ...`).
            # So if it returns False, the state should be clean.
            # Let's verify: 
            # `used_diffs` is a set. `used_diffs -= current_diffs` removes elements.
            # If we added them, we remove them. Correct.
            
            if backtrack(0):
                return rows
        return None

    # Iterate M from lower bound upwards
    # Start from M_min, but jump to a reasonable estimate if M_min is too low and search is slow?
    # For (7,5), M_min=14 is wrong? 
    # Wait, the counting bound M*(M+1)/2 >= 105 gives M=14.
    # But the max entry is M. The differences are in 1..M.
    # Is it possible to have 105 distinct differences in 1..14? No, only 14*15/2 = 105 differences possible.
    # So M=14 is the theoretical minimum if we could pack them perfectly.
    # But the structure (rows starting at 0) imposes constraints.
    # The known best is 112.
    # So we should start M around 105-115.
    
    # Let's start from max(M_min, 100) to save time, or just M_min if we want to be rigorous.
    # Given 120s, and backtracking can be exponential, starting too low might waste time.
    # Let's start from 105.
    
    M_start = max(M_min, 100)
    
    for M in range(M_start, 150):  # Upper bound 150 is safe
        if time.time() - start_time > time_limit:
            break
            
        # Try to find solution with max entry <= M
        # Use multiple random restarts
        found = None
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            # We need a fresh state for each attempt
            # The `search_rows` function above has closure issues with `rows` and `used_diffs` if not reset.
            # Let's rewrite `search_rows` to take M and seed, and manage its own state.
            
            # Actually, let's just implement the search inline or with a helper that takes state.
            
            pass # Placeholder, will replace with proper implementation

    # The above logic is getting complex with closures. Let's write a cleaner version.
    
    def solve_for_M(M, seed):
        rng = random.Random(seed)
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx):
            if time.time() - start_time > time_limit:
                return False
            if row_idx == n_rows:
                return True
                
            row = [0]
            
            def build(pos, cur_row, cur_diffs):
                if time.time() - start_time > time_limit:
                    return False
                if pos == k + 1:
                    if cur_diffs & used_diffs:
                        return False
                    rows.append(list(cur_row))
                    used_diffs.update(cur_diffs)
                    if backtrack(row_idx + 1):
                        return True
                    rows.pop()
                    used_diffs -= cur_diffs
                    return False
                    
                start_val = cur_row[-1] + 1
                end_val = M
                candidates = list(range(start_val, end_val + 1))
                rng.shuffle(candidates)
                
                # Heuristic: try smaller values first? Or random?
                # Random is good for diversity.
                for x in candidates:
                    new_diffs = {x - a for a in cur_row}
                    if len(new_diffs) != len(cur_row):
                        continue
                    if new_diffs & cur_diffs:
                        continue
                    if new_diffs & used_diffs:
                        continue
                        
                    cur_row.append(x)
                    new_cur_diffs = cur_diffs | new_diffs
                    
                    if build(pos + 1, cur_row, new_cur_diffs):
                        return True
                    cur_row.pop()
                return False
            
            return build(1, row, set())
        
        return backtrack(0), rows

    best_rows = None
    best_M = float('inf')
    
    # Try M values
    # Start from a slightly higher bound to ensure we find a solution quickly, 
    # then maybe try to lower it? 
    # Or just start from 105 and go up.
    
    for M in range(105, 120):
        if time.time() - start_time > time_limit:
            break
            
        # Try several seeds for this M
        for seed in range(200):
            if time.time() - start_time > time_limit:
                break
                
            success, rows = solve_for_M(M, seed)
            if success:
                best_rows = rows
                best_M = M
                break # Found a solution for this M, move to next M? 
                # No, we want the smallest M. So if we find one for M, 
                # we should check if we can find one for M-1? 
                # We are iterating M upwards. So the first M we find is the smallest 
                # *among those we tried*. 
                # If we start at 105, and find a solution, that's the best possible 
                # (if 105 is the true min). If we don't find it in 200 seeds, 
                # we might miss it. 
                # So we should probably stick with one M for a long time?
                # Or iterate M, and for each M, try many seeds.
                # If we find a solution for M=112, we return it.
                # If we don't find for 105..111, we return 112.
                
                break # Break seed loop, go to next M? 
                # Actually, if we found a solution for M, we can return immediately 
                # if we are confident it's the minimum. 
                # But since we start from the lower bound, the first success is the min 
                # *for the seeds we tried*. 
                # To be safe, we should return the first valid solution found 
                # as we iterate M from low to high.
                
                return rows
        
    # If no solution found in the loop, return a greedy solution as fallback
    # (Though the loop should find one)
    
    # Fallback: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used) and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

