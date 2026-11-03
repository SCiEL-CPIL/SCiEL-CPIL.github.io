import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    # Safety margin: 120s limit, use 115s for search
    DEADLINE = t0 + 115.0
    row_len = k + 1
    rng = random.Random(42)
    
    # Pre-compute pairs for a row to speed up difference calculation
    row_pairs = [(i, j) for i in range(row_len) for j in range(i+1, row_len)]
    
    def get_row_diffs(row):
        """Calculate all positive differences for a single row."""
        return [row[j] - row[i] for i, j in row_pairs]
    
    def get_scope(rows):
        """Calculate the maximum entry in all rows."""
        # Rows are increasing, so max is the last element
        return max(row[-1] for row in rows)
    
    def count_conflicts(counter):
        """Count the number of differences that appear more than once."""
        # A conflict exists for every duplicate occurrence.
        # If a diff appears 3 times, it contributes 2 to the conflict count.
        # Total conflicts = sum(count - 1 for count in counts if count > 1)
        # This is equivalent to (total_diffs - unique_diffs)
        # But we want the penalty to reflect the "badness". 
        # Let's use the number of "extra" occurrences.
        return sum(c - 1 for c in counter.values() if c > 1)
    
    def greedy_initial():
        """Generate a valid initial solution using greedy."""
        matrix = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                last += 1
                new_diffs = {last - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last)
                    used_diffs |= new_diffs
            matrix.append(row)
        return matrix

    def build_row_avoiding(used_diffs, cap, jitter=0.0):
        """
        Build a new row of length row_len starting at 0,
        avoiding differences in used_diffs.
        cap is the maximum allowed value for the last element.
        Returns None if not possible within cap.
        """
        row = [0]
        current_max = 0
        attempts = 0
        max_attempts = 1000 # Prevent infinite loop
        
        while len(row) < row_len:
            if attempts > max_attempts:
                return None
            attempts += 1
            
            # Try to find the next smallest integer that doesn't create a conflict
            # We start from current_max + 1
            # But to allow more flexibility, we can scan upwards
            
            # Optimization: Instead of checking every integer, we can just try the next one,
            # and if it fails, increment. Since we are greedy, we want small values.
            
            next_val = current_max + 1
            found = False
            
            # We might need to skip several values
            # To speed up, we check conflicts
            while next_val <= cap:
                new_diffs = {next_val - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(next_val)
                    current_max = next_val
                    used_diffs = used_diffs | new_diffs
                    found = True
                    break
                next_val += 1
            
            if not found:
                return None
                
            # Optional jitter: randomly skip a value to diversify
            # This is handled by the caller if needed, but for greedy rebuild,
            # we usually want the tightest fit.
            
        return row

    # 1. Initial Solution
    current_matrix = greedy_initial()
    best_matrix = [r[:] for r in current_matrix]
    best_scope = get_scope(best_matrix)
    
    # Initialize difference counter for current state
    diff_counter = Counter()
    for row in current_matrix:
        for d in get_row_diffs(row):
            diff_counter[d] += 1
            
    current_conflicts = count_conflicts(diff_counter)
    current_scope = get_scope(current_matrix)
    current_cost = current_scope + 1000 * current_conflicts
    
    # 2. Simulated Annealing / Local Search
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() < DEADLINE:
        # Choose move type
        # 70% Single Entry Shift, 30% Row Rebuild
        move_type = rng.random()
        r_idx = rng.randint(0, n_rows - 1)
        
        if move_type < 0.7:
            # --- Move Type 1: Shift a single entry ---
            # Pick a random index > 0
            i_idx = rng.randint(1, row_len - 1)
            
            old_val = current_matrix[r_idx][i_idx]
            
            # Determine valid range for new_val
            lo = current_matrix[r_idx][i_idx-1] + 1
            # If it's the last element, hi can be large (to allow expansion if needed to escape)
            # But we want to minimize scope, so bias towards smaller.
            if i_idx < row_len - 1:
                hi = current_matrix[r_idx][i_idx+1] - 1
            else:
                # Allow increasing the max slightly to escape local minima
                hi = current_scope + 10 
            
            if lo > hi:
                # No valid integer in the gap? 
                # This can happen if neighbors are too close (gap < 2).
                # In that case, we can't move this element without moving neighbors.
                # Skip this move.
                continue
            
            # Propose new value
            # Bias towards lower values to reduce scope
            # But allow some randomness
            if rng.random() < 0.7:
                # Try to reduce
                new_val = rng.randint(lo, min(old_val, hi))
            else:
                # Try random in range
                new_val = rng.randint(lo, hi)
                
            if new_val == old_val:
                continue
            
            # Check if the new value maintains strict increasing order
            # (Already checked by lo/hi logic, but double check for safety)
            if i_idx > 0 and new_val <= current_matrix[r_idx][i_idx-1]:
                continue
            if i_idx < row_len - 1 and new_val >= current_matrix[r_idx][i_idx+1]:
                continue
                
            # Apply move
            old_row = current_matrix[r_idx]
            old_diffs = get_row_diffs(old_row)
            
            # Remove old diffs from counter
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            # Update row
            new_row = old_row[:]
            new_row[i_idx] = new_val
            
            # Add new diffs to counter
            new_diffs = get_row_diffs(new_row)
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
                
            # Calculate new cost
            new_conflicts = count_conflicts(diff_counter)
            new_scope = get_scope(current_matrix) # Scope might change if i_idx is last
            # Optimization: Only recalc scope if i_idx is last
            if i_idx == row_len - 1:
                new_scope = new_row[-1] if new_row[-1] > new_scope else new_scope
                # Actually, if we reduced the last element, scope might decrease.
                # If we increased it, scope might increase.
                # get_scope is O(n_rows), which is cheap (7).
                new_scope = get_scope(current_matrix)
            
            new_cost = new_scope + 1000 * new_conflicts
            delta = new_cost - current_cost
            
            # Acceptance criterion
            if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                # Accept
                current_matrix[r_idx] = new_row
                current_conflicts = new_conflicts
                current_scope = new_scope
                current_cost = new_cost
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_matrix = [r[:] for r in current_matrix]
            else:
                # Revert
                # Restore old diffs
                for d in new_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                # Restore new (old) diffs
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                # Restore row
                current_matrix[r_idx] = old_row
                
        else:
            # --- Move Type 2: Rebuild Row ---
            old_row = current_matrix[r_idx]
            old_diffs = get_row_diffs(old_row)
            
            # Remove old diffs
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            # Get current used diffs (keys of counter)
            used_diffs = set(diff_counter.keys())
            
            # Try to build a new row
            # Cap: Try to fit within current scope, or slightly larger
            cap = current_scope + rng.randint(0, 5)
            new_row = build_row_avoiding(used_diffs, cap)
            
            if new_row is None:
                # Restore old row
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                # Skip
                continue
            
            # Add new diffs
            new_diffs = get_row_diffs(new_row)
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
                
            # Calculate new cost
            new_conflicts = count_conflicts(diff_counter)
            new_scope = get_scope(current_matrix)
            new_cost = new_scope + 1000 * new_conflicts
            delta = new_cost - current_cost
            
            if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                # Accept
                current_matrix[r_idx] = new_row
                current_conflicts = new_conflicts
                current_scope = new_scope
                current_cost = new_cost
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_matrix = [r[:] for r in current_matrix]
            else:
                # Revert
                for d in new_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                current_matrix[r_idx] = old_row

        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T

    return best_matrix
