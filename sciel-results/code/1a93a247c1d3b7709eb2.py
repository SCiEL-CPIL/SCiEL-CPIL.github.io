import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    timeout = 115.0
    
    best_rows = None
    best_max = float('inf')
    
    # Use a list to track best solution
    # We will use a recursive backtracking search with pruning.
    
    # To prevent stack overflow or deep recursion issues in Python,
    # we might need to increase recursion limit, but n_rows=7, k=6 is small.
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def search(row_idx, current_rows, used_diffs, current_max):
        nonlocal best_rows, best_max
        
        if time.time() - start_time > timeout:
            return

        if row_idx == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
            return

        # Pruning: If we already have a solution with max < current_max,
        # and we are building a row, if the last element of the current row 
        # (once finished) is likely to be >= best_max, we might prune.
        # But we don't know the last element yet.
        # We can prune if the *minimum possible* max for remaining rows is >= best_max.
        # This is hard to estimate. We'll rely on the greedy nature to keep max low.
        
        # Construct the next row
        # We build the row element by element.
        # row is a list, we extend it.
        
        def build_row(pos, row, row_used_diffs):
            nonlocal used_diffs, current_max, best_max, best_rows
            
            if time.time() - start_time > timeout:
                return False

            if pos == k + 1:
                # Row complete
                # Update global used_diffs
                # The differences are in row_used_diffs
                # Check if this row's max is within bounds
                last_val = row[-1]
                
                # Recurse to next row
                current_rows.append(row)
                # Merge diffs
                # Note: row_used_diffs is the set of diffs for this row.
                # We need to pass the union of used_diffs and row_used_diffs to the next call.
                # But used_diffs is a set. We can just pass a new set or modify and revert.
                # For simplicity and safety in recursion, let's pass a new set? 
                # That's expensive. Let's use a list of sets or a global set with backtracking.
                
                # Let's use a global `active_used` set and add/remove.
                # But `search` is called with `used_diffs`. 
                # Let's change strategy: 
                # `search` takes `used_diffs` as an argument. 
                # `build_row` is inner, so it can modify a local copy or use a global mutable structure.
                
                # Let's use a global `global_used` set and manage it in the outer loop?
                # No, let's stick to passing sets. Since k=6, row has 21 diffs. 
                # Creating a new set of size ~100 is fast.
                
                next_used = used_diffs | row_used_diffs
                next_max = max(current_max, last_val)
                
                # Pruning: if next_max >= best_max, no point going deeper?
                # Actually, if next_max == best_max, we might find a different config but same max.
                # If we want strictly better, we prune if next_max >= best_max.
                # But we want to find *any* solution if best_max is inf.
                if next_max < best_max:
                    search(row_idx + 1, current_rows, next_used, next_max)
                
                current_rows.pop()
                return True

            # Try to pick next element
            # Heuristic: Try smallest valid x first
            # x must be > row[-1]
            # Diffs: x - a for a in row
            # These diffs must not be in row_used_diffs (internal collision)
            # And must not be in used_diffs (global collision)
            
            start_x = row[-1] + 1
            
            # Optimization: If we already have a best_max, we can limit x
            # The max value in this row cannot exceed best_max - 1 (if we want to improve)
            # Actually, if current_max is 100 and best_max is 105, x can be up to 104.
            # If current_max is 0, x can be anything, but we want to keep it small.
            
            # Let's iterate x from start_x upwards.
            # To avoid infinite loop, we can set a cap.
            # Cap = best_max if best_max < inf else 200 (heuristic cap)
            
            cap = best_max if best_max < float('inf') else 200
            # If we are building the first row, and best_max is inf, we just go greedy.
            
            # We can also randomize the start or order if we want, 
            # but "from scratch" implies a deterministic or structured approach.
            # Let's try a few random restarts if it gets stuck?
            # For now, pure greedy backtracking.
            
            for x in range(start_x, cap + 1):
                # Check collisions
                new_diffs = {x - a for a in row}
                
                # Check internal
                if new_diffs & row_used_diffs:
                    continue
                
                # Check global
                if new_diffs & used_diffs:
                    continue
                
                # Valid
                row.append(x)
                new_row_used = row_used_diffs | new_diffs
                
                # Recurse
                if build_row(pos + 1, row, new_row_used):
                    # If we found a solution deeper, we might want to stop?
                    # No, we want to find the *best* (min max).
                    # But if we found a solution, we can prune other branches 
                    # if we are just looking for *a* solution.
                    # Since we are looking for min max, we must continue searching
                    # for smaller max values.
                    # However, if we found a solution with max M, 
                    # any branch that leads to max >= M can be pruned.
                    # This is handled by the `next_max < best_max` check in the parent.
                    
                    # We should NOT return True immediately to stop the search,
                    # unless we are satisfied. 
                    # But wait, if we return True, the caller might stop?
                    # In my structure, `build_row` returns True if it *successfully* built a row and recursed.
                    # It doesn't mean "stop all search".
                    # The caller `search` will continue its loop?
                    # No, `build_row` is called once per `search` call to generate *one* row?
                    # No, `build_row` explores all possibilities for the current row.
                    # So `build_row` should return when it has exhausted all valid rows 
                    # for this position, or when it finds a solution that updates best_max?
                    # Actually, if we find a solution, we update best_max.
                    # Then subsequent branches in `build_row` will be pruned by the cap.
                    
                    # So we just let it run.
                    pass
                    
                row.pop()
                
            return True

        # Start building row 0
        # Initial row is [0]
        # Initial diffs is empty set
        build_row(1, [0], set())

    # Run search
    # To handle cases where the deterministic greedy path is a local minimum,
    # we can run multiple searches with slight perturbations (randomized greedy).
    # But let's start with the pure deterministic one.
    
    search(0, [], set(), 0)
    
    if best_rows is None:
        # Fallback: Greedy construction (like the seed but simpler)
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
        
    return best_rows
