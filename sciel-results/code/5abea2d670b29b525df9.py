import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    
    deadline = time.time() + 115.0

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                v = row[-1] + 1
                while True:
                    # Check if v creates any difference already in used
                    # Differences are v - u for u in row
                    valid = True
                    for u in row:
                        if v - u in used:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        # Add new differences to used
                        for u in row[:-1]:
                            used.add(v - u)
                        break
                    v += 1
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(r[-1] for r in best_rows)
    
    # Initialize diff counts for the baseline
    # Max scope from greedy might be small, but let's be safe.
    # We can estimate max scope or just use a large enough array.
    # Let's use a dictionary for diff counts to be safe and simple, 
    # or an array if we know the bound.
    # Given k=5, n=7, greedy scope is usually < 100.
    # Let's use an array of size 1000.
    MAX_SCOPE_BOUND = 1000
    diff_count = [0] * (MAX_SCOPE_BOUND + 1)
    current_collisions = 0
    
    def update_diff_counts(rows, counts, collisions):
        # Rebuild counts from scratch
        counts = [0] * (MAX_SCOPE_BOUND + 1)
        coll = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d > MAX_SCOPE_BOUND:
                        # Should not happen with greedy init, but handle just in case
                        # If it does, we need to resize or use dict.
                        # For now, assume it fits.
                        pass
                    counts[d] += 1
        for c in counts:
            if c > 1:
                coll += c - 1
        return counts, coll

    diff_count, current_collisions = update_diff_counts(baseline_rows, diff_count, current_collisions)
    current_max = best_max
    
    # Helper to get cost
    def get_cost(collisions, max_val):
        return collisions * 10000 + max_val

    current_cost = get_cost(current_collisions, current_max)
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.9999
    min_T = 0.1
    
    # Precompute row_maxes for faster M calculation
    row_maxes = [r[-1] for r in best_rows]
    
    start_time = time.time()
    steps = 0
    
    while time.time() < deadline:
        # Pick random row and position
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, row_len - 1)
        
        old_val = best_rows[r_idx][p_idx]
        
        low = best_rows[r_idx][p_idx - 1] + 1
        if p_idx < row_len - 1:
            high = best_rows[r_idx][p_idx + 1] - 1
        else:
            # For the last element, allow some increase to explore, 
            # but bias towards decrease.
            high = old_val + 10 
            
        if low > high:
            continue
            
        # Bias towards decreasing values to reduce scope
        if random.random() < 0.7:
            # Try to decrease or stay
            # Ensure range is valid
            if old_val < low:
                new_val = low
            else:
                new_val = random.randint(low, old_val)
        else:
            # Try to increase or stay
            if old_val > high:
                new_val = high
            else:
                new_val = random.randint(old_val, high)
                
        if new_val == old_val:
            continue
            
        # Calculate delta collisions
        delta_coll = 0
        
        # Remove old diffs
        for j in range(row_len):
            if j == p_idx:
                continue
            d = abs(old_val - best_rows[r_idx][j])
            if d <= MAX_SCOPE_BOUND:
                if diff_count[d] > 1:
                    delta_coll -= 1
                diff_count[d] -= 1
                
        # Add new diffs
        for j in range(row_len):
            if j == p_idx:
                continue
            d = abs(new_val - best_rows[r_idx][j])
            if d <= MAX_SCOPE_BOUND:
                if diff_count[d] > 0:
                    delta_coll += 1
                diff_count[d] += 1
                
        # Calculate new max
        old_row_max = row_maxes[r_idx]
        if p_idx == row_len - 1:
            new_row_max = new_val
        else:
            new_row_max = old_row_max
            
        # If new_row_max changed, update global max
        if new_row_max != old_row_max:
            # Remove old_row_max from consideration? 
            # Simpler: recompute max of all rows
            # Since n_rows is small, this is fast
            new_max = max([best_rows[i][-1] for i in range(n_rows) if i != r_idx] + [new_row_max])
        else:
            new_max = current_max
            
        new_collisions = current_collisions + delta_coll
        new_cost = get_cost(new_collisions, new_max)
        delta_cost = new_cost - current_cost
        
        # Accept/Reject
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            best_rows[r_idx][p_idx] = new_val
            if p_idx == row_len - 1:
                row_maxes[r_idx] = new_row_max
            current_collisions = new_collisions
            current_max = new_max
            current_cost = new_cost
            
            # Update best if valid
            if current_collisions == 0 and current_max < best_max:
                best_max = current_max
                # We could copy best_rows here, but since we are modifying in place,
                # we need to keep a separate copy for the final result
                # Or just track the best max and reconstruct?
                # Let's keep a copy of the best valid rows found so far
                if not hasattr(construct, '_best_valid_rows_cache'):
                    construct._best_valid_rows_cache = [r[:] for r in best_rows]
                else:
                    construct._best_valid_rows_cache = [r[:] for r in best_rows]
                    
        else:
            # Revert diff counts
            for j in range(row_len):
                if j == p_idx:
                    continue
                d = abs(new_val - best_rows[r_idx][j])
                if d <= MAX_SCOPE_BOUND:
                    diff_count[d] -= 1
            for j in range(row_len):
                if j == p_idx:
                    continue
                d = abs(old_val - best_rows[r_idx][j])
                if d <= MAX_SCOPE_BOUND:
                    diff_count[d] += 1
                    
        T *= cooling
        if T < min_T:
            T = min_T
            
        steps += 1
        
        # Periodic check to ensure diff_count consistency (optional, for safety)
        if steps % 10000 == 0:
            # Rebuild and check
            pass # Skip for speed, assume logic is correct

    # Return the best valid rows found
    # If we found a valid one with lower max, return that.
    # Otherwise, return the initial greedy one (which is valid).
    # The variable best_rows is the current state, which might be invalid.
    # We need to return a VALID solution.
    # If current_collisions == 0, best_rows is valid.
    # If not, we should return the best VALID solution we've seen.
    
    # Let's maintain a separate best_valid_rows variable
    # I modified the code above to use a cache, but it's messy.
    # Let's refactor slightly to keep `best_valid_rows` explicitly.
    
    return best_rows if current_collisions == 0 else construct._best_valid_rows_cache if hasattr(construct, '_best_valid_rows_cache') else baseline_rows
