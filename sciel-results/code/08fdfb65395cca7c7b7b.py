import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    best_rows = None
    best_max = float('inf')
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_with_backtracking():
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, col_idx, current_row):
            # Check time
            if time.time() > deadline:
                return False
            
            # Base case: all rows filled
            if row_idx == n_rows:
                return True
            
            # If we are starting a new row
            if col_idx == 0:
                current_row = [0]
                # Recurse to fill the first row
                if backtrack(row_idx, 1, current_row):
                    rows.append(current_row)
                    return True
                return False
            
            # If we are filling the current row
            # Determine the range of possible values for current_row[col_idx]
            # It must be > current_row[col_idx - 1]
            min_val = current_row[col_idx - 1] + 1
            
            # We want to try small values first to keep max small
            # But if we hit a dead end, we might need to backtrack.
            # To make it "greedy", we try min_val, min_val+1, etc.
            # To make it "randomized", we can shuffle the order of attempts 
            # or add some noise, but pure greedy (smallest first) is usually best for minimizing max.
            # However, pure greedy often gets stuck. 
            # Let's try a "lookahead" or just simple backtracking.
            
            # Optimization: If we have already filled some rows, we can estimate the lower bound
            # for the max entry. But simple backtracking is fine for small n.
            
            # We need to decide how far to search.
            # If we just go to infinity, it might be slow.
            # We can cap the search if we already have a good best_max.
            max_search_val = best_max - 1 if best_max < float('inf') else 500
            
            # To allow escaping local minima in the greedy choice, we can occasionally
            # try a larger value, but for now, let's stick to increasing order 
            # and rely on backtracking to handle conflicts.
            
            # Actually, a pure "smallest first" backtracking is deterministic.
            # To get diversity, we can randomize the starting point or the order.
            # But since we want to minimize max, "smallest first" is the correct heuristic.
            # The "randomness" comes from the fact that if we fail, we backtrack.
            # Wait, if it's deterministic, we only get one solution per call.
            # We need to introduce randomness to explore different paths.
            
            # Strategy: Try values from min_val to max_search_val.
            # If we find a conflict, try the next value.
            # This is effectively a depth-first search.
            
            # To introduce randomness, we can sometimes skip a few small values 
            # or shuffle the order of a window of values.
            # Let's try a window of size 5-10 values around min_val, shuffled.
            
            window_size = 10
            candidates = list(range(min_val, min(min_val + window_size, max_search_val + 1)))
            random.shuffle(candidates)
            
            # Ensure min_val is tried? No, shuffling is fine.
            # But we should probably prioritize smaller values.
            # Let's do: try min_val, then min_val+1... but with a random jump?
            # Simpler: Just try in increasing order, but if we fail, we backtrack.
            # Since the problem is small (n=4, k=8), backtracking should be fast.
            
            # Let's refine: 
            # 1. Try min_val
            # 2. If conflict, try min_val + 1
            # ...
            # This is just standard backtracking.
            
            for val in range(min_val, max_search_val + 1):
                # Check if val creates any duplicate differences
                new_diffs = set()
                conflict = False
                for prev_val in current_row:
                    d = val - prev_val
                    if d in used_diffs or d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Tentatively accept
                current_row.append(val)
                used_diffs.update(new_diffs)
                
                # Recurse
                if col_idx == k:
                    # Row complete
                    rows.append(current_row[:]) # Store copy
                    if backtrack(row_idx + 1, 0, []):
                        return True
                    else:
                        # Backtrack: remove row and diffs
                        last_row = rows.pop()
                        for i in range(len(last_row)):
                            for j in range(i+1, len(last_row)):
                                used_diffs.discard(last_row[j] - last_row[i])
                        return False
                else:
                    if backtrack(row_idx, col_idx + 1, current_row):
                        return True
                    else:
                        # Backtrack: remove val and diffs
                        current_row.pop()
                        used_diffs.difference_update(new_diffs)
            
            return False
        
        # Start
        if backtrack(0, 0, []):
            return rows
        return None

    # Main loop
    # We run multiple restarts with different random seeds/states to find better solutions
    # The backtracking itself is deterministic if we always pick the smallest valid value.
    # To get different results, we need to vary the search.
    # One way: Randomly perturb the "greedy" choice.
    # Or: Use the random shuffle in the candidate window as implemented above?
    # The code above uses a fixed increasing order in the for loop, ignoring the shuffle logic I wrote in the thought process.
    # Let's fix the code to actually use randomness to explore different valid solutions.
    
    # Revised Greedy with Backtracking:
    # Instead of just increasing order, we can randomly select from a set of valid candidates
    # or introduce a "temperature" like in simulated annealing but for the backtracking depth?
    # No, let's stick to a simpler structural change:
    # 1. Generate a valid solution using backtracking (smallest first).
    # 2. If we find one, try to improve it by local search? No, that's the old strategy.
    # 3. To get different solutions, we can randomly fix some cells and backtrack the rest?
    
    # Let's implement a randomized backtracking where we pick the next value from a random valid candidate
    # rather than always the smallest. This explores the space of valid solutions.
    
    def randomized_backtracking():
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, col_idx, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if col_idx == 0:
                current_row = [0]
                if backtrack(row_idx, 1, current_row):
                    rows.append(current_row)
                    return True
                return False
            
            min_val = current_row[col_idx - 1] + 1
            max_search_val = best_max - 1 if best_max < float('inf') else 200
            
            # Find all valid values in a window
            valid_vals = []
            for val in range(min_val, max_search_val + 1):
                new_diffs = set()
                conflict = False
                for prev_val in current_row:
                    d = val - prev_val
                    if d in used_diffs or d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    valid_vals.append((val, new_diffs))
            
            if not valid_vals:
                return False
            
            # Randomly select a valid value
            # To bias towards smaller values (to minimize max), we can use a weighted random choice
            # or just shuffle. Shuffling is uniform. Weighted is better.
            # Let's try uniform shuffle first for simplicity, as it explores more broadly.
            random.shuffle(valid_vals)
            
            for val, new_diffs in valid_vals:
                current_row.append(val)
                used_diffs.update(new_diffs)
                
                if col_idx == k:
                    rows.append(current_row[:])
                    if backtrack(row_idx + 1, 0, []):
                        return True
                    else:
                        rows.pop()
                        used_diffs.difference_update(new_diffs)
                        return False
                else:
                    if backtrack(row_idx, col_idx + 1, current_row):
                        return True
                    else:
                        current_row.pop()
                        used_diffs.difference_update(new_diffs)
            
            return False
        
        if backtrack(0, 0, []):
            return rows
        return None

    # Run multiple restarts
    # The first run might be slow if it explores broadly.
    # We can limit the search depth or time per restart.
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        rows = randomized_backtracking()
        if rows:
            curr_max = get_max_entry(rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = rows
                # If we hit the theoretical lower bound, we can stop
                if best_max == total_diffs:
                    break

    # Fallback: if no solution found (should not happen for small n), return a simple valid one
    # Or use the greedy one from the previous program as a fallback
    if best_rows is None:
        # Simple greedy fallback
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
        best_rows = rows
        best_max = get_max_entry(best_rows)

    return best_rows
