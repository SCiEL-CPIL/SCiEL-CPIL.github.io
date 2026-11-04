import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # k is the number of elements after 0. So total elements = k + 1.
    # Task says (7,5) but interface is construct(n_rows, k).
    # Prompt: "Each row is a list of k + 1 integers".
    # If title is (7,5), maybe k=5?
    # "Choose 7 rows of 6 increasing integers starting at 0" -> 6 integers total?
    # If 6 integers total, then k=5 (since k+1=6).
    # Differences: C(6,2) = 15. Total diffs = 7*15 = 105.
    # This matches the "counting bound 105".
    # So we assume the row has k+1 elements.
    
    row_size = k + 1
    diffs_per_row = row_size * (row_size - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for scope M
    # The differences are 1..M. We need total_diffs distinct differences.
    # So M >= total_diffs.
    lower_bound = total_diffs
    
    # Best known is 112.
    # We will search for M starting from lower_bound.
    
    best_rows = None
    best_scope = float('inf')
    
    # Fallback greedy
    def deterministic_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_size:
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used.update(new_diffs)
                        last = x
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = deterministic_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Backtracking search
    def search_for_scope(M):
        """
        Returns a valid set of rows with scope <= M, or None.
        Uses backtracking.
        """
        used_diffs = set()
        rows = []
        current_row = [0]
        
        # We need to fill n_rows rows.
        # State: (row_idx, current_row)
        
        def backtrack(row_idx, curr_row, last_val):
            if time.time() - start_time > deadline:
                return False, None
            
            if row_idx == n_rows:
                return True, [r[:] for r in rows]
            
            # If we are starting a new row
            if len(curr_row) == 1:
                # The first element is 0.
                # We need to pick the next element.
                pass # Handled in the loop below
            
            # Try to extend current_row
            # Candidates for next element x
            # x must be > last_val
            # x must be <= M
            # New diffs x - a for a in curr_row must not be in used_diffs
            
            # Heuristic: Try smaller x first to keep scope low?
            # But if we pick too small, we might run out of room for later elements?
            # Actually, smaller x is generally better for minimizing max.
            
            # Optimization:
            # If curr_row is [0, a1, ...], the next element x creates diffs x, x-a1, ...
            
            # We can prune if even the smallest possible future diffs exceed M?
            # Not strictly, but we can check if we can complete the row.
            
            # Let's iterate x from last_val + 1 to M
            # To speed up, we can limit the range.
            # The remaining elements in the row are row_size - len(curr_row).
            # The smallest possible next element is last_val + 1.
            # The largest possible element in the row is M.
            
            # We'll just iterate.
            for x in range(last_val + 1, M + 1):
                new_diffs = []
                valid = True
                for a in curr_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check if we can complete the rest of the row?
                # Simple heuristic: If we have few elements left, check if space allows.
                # For now, just recurse.
                
                # Add to state
                curr_row.append(x)
                used_diffs.update(new_diffs)
                
                if len(curr_row) == row_size:
                    # Row complete
                    rows.append(curr_row[:])
                    # Start next row
                    found, res = backtrack(row_idx + 1, [0], 0)
                    if found:
                        return True, res
                    rows.pop()
                    # Backtrack
                    curr_row.pop()
                    used_diffs.difference_update(new_diffs)
                else:
                    # Continue current row
                    found, res = backtrack(row_idx, curr_row, x)
                    if found:
                        return True, res
                    # Backtrack
                    curr_row.pop()
                    used_diffs.difference_update(new_diffs)
            
            return False, None
        
        # Start search
        found, res = backtrack(0, [0], 0)
        if found:
            return res
        return None

    # Try to find solutions for increasing M
    # We start from lower_bound.
    # If we find a solution for M, it's the best possible for that M.
    # We want the smallest M.
    
    # To use time effectively:
    # 1. Try M = lower_bound. If found, return.
    # 2. If not, try M = lower_bound + 1, etc.
    # But backtracking can be slow.
    # We can add randomness to the candidate order to escape dead ends?
    # Or just run multiple restarts for the same M?
    
    # Strategy:
    # Allocate time chunks to different M values.
    # Since 112 is known, we know a solution exists for 112.
    # We should definitely try to beat 112.
    
    # Let's try M from lower_bound to best_scope - 1.
    # If we find a solution with M < best_scope, update best.
    # We can interleave searches.
    
    # For simplicity and robustness:
    # 1. Try M = lower_bound to lower_bound + 5 with a short time limit each.
    # 2. If none found, try M = lower_bound + 5 to best_scope - 1.
    
    # Actually, just iterate M from lower_bound upwards, with a time limit per M.
    # If we find a solution, return it immediately (since we are going up from the bound, the first found is the minimum *for this search path*, but we should ensure we don't miss a smaller one? No, we start from the absolute lower bound. If we find M=105, it's optimal. If we fail 105, we try 106. If we find 106, it's better than 112. We can stop there?
    # Yes, if we find 106, it's better than 112. We don't need to check 105 again.
    # So we can return the first solution found when iterating M from lower_bound upwards.
    
    # However, backtracking might fail for 105 due to time limit, but a solution might exist.
    # We should be careful.
    
    # Let's add a randomized component to the backtracking candidate order.
    # Instead of always trying smallest x first, we can shuffle candidates slightly?
    # No, smallest x is best for minimizing max.
    # But if we get stuck, we might need to try larger x earlier.
    
    # Modified Backtracker with Randomization:
    def search_for_scope_randomized(M, time_limit=5.0):
        start_search = time.time()
        
        # To avoid getting stuck in the same dead end, we can randomize the order of rows?
        # Or randomize the candidate selection?
        
        # Let's stick to deterministic smallest-first, but add a timeout.
        # If it times out, we try again with a different random seed for row ordering?
        
        # Simple approach: Run the deterministic backtracker.
        # If it finds a solution, return it.
        # If it times out, return None.
        
        # To make it more robust, we can try a few random restarts of the backtracker
        # where we slightly perturb the candidate order?
        
        # For now, let's just use the deterministic one, but with a tight time limit per M.
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, curr_row, last_val):
            if time.time() - start_search > time_limit:
                return False, None
            
            if row_idx == n_rows:
                return True, [r[:] for r in rows]
            
            for x in range(last_val + 1, M + 1):
                new_diffs = []
                valid = True
                for a in curr_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                curr_row.append(x)
                used_diffs.update(new_diffs)
                
                if len(curr_row) == row_size:
                    rows.append(curr_row[:])
                    found, res = backtrack(row_idx + 1, [0], 0)
                    if found:
                        return True, res
                    rows.pop()
                    curr_row.pop()
                    used_diffs.difference_update(new_diffs)
                else:
                    found, res = backtrack(row_idx, curr_row, x)
                    if found:
                        return True, res
                    curr_row.pop()
                    used_diffs.difference_update(new_diffs)
            
            return False, None
        
        found, res = backtrack(0, [0], 0)
        if found:
            return res
        return None

    # Search loop
    # We try M from lower_bound up to best_scope - 1.
    # We give each M a time budget.
    # Total time available: deadline - (time.time() - start_time)
    
    current_time = time.time()
    time_per_m = 2.0  # seconds
    
    for M in range(lower_bound, best_scope):
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Try to find a solution for this M
        # We can try multiple times if time permits?
        # For now, one attempt per M.
        
        res = search_for_scope_randomized(M, time_limit=time_per_m)
        
        if res:
            best_rows = res
            best_scope = M
            # If we found the lower bound, we can't do better.
            if M == lower_bound:
                break
            # If we found a solution, it's the best possible for this M.
            # Since we are iterating upwards, this is the smallest M we found.
            # We should stop?
            # Yes, because we started from the lower bound.
            # If we found M=106, it's better than 112.
            # We don't need to check 105 again.
            break

    return best_rows
