import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave margin for 120s limit
    
    # 1. Greedy Initialization
    # We need a valid starting point.
    # Greedy: For each row, pick the smallest possible next integer that doesn't conflict.
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We need k more elements.
            # To keep the max small, we should try small values first.
            # But simple greedy might get stuck or produce a large max.
            # Let's use a randomized greedy to allow restarts if needed, 
            # but for SA, one good valid start is enough.
            for _ in range(k):
                # Try values starting from row[-1] + 1
                found = False
                # Upper bound for search in greedy: let's be generous, say 200
                # But we want to minimize max, so we should start low.
                for x in range(row[-1] + 1, 200):
                    new_diffs = {x - a for a in row}
                    # Check internal uniqueness
                    if len(new_diffs) != len(row):
                        continue
                    # Check global uniqueness
                    if new_diffs & used:
                        continue
                    # Valid
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    # If we can't find a small one, expand range
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) != len(row) and not (new_diffs & used):
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        # Fallback: just pick a large random one to ensure validity, then SA will fix it
                        # This is rare
                        x = row[-1] + 100
                        while True:
                            new_diffs = {x - a for a in row}
                            if len(new_diffs) == len(row) and not (new_diffs & used):
                                row.append(x)
                                used.update(new_diffs)
                                break
                            x += 1
            rows.append(row)
        return rows, used

    initial_rows, initial_used = greedy_init()
    current_rows = initial_rows
    current_used = initial_used
    current_max = max(max(r) for r in current_rows)
    best_rows = [list(r) for r in current_rows]
    best_max = current_max
    
    # Helper to get diffs of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check if a row is internally valid (all diffs unique)
    def is_row_valid(row):
        diffs = get_diffs(row)
        return len(diffs) == (len(row) * (len(row) - 1)) // 2

    # Simulated Annealing
    rng = random.Random(42)
    temperature = 20.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        if temperature < 0.01:
            break
            
        # Pick a random row to mutate
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[row_idx]
        
        # Remove old row's diffs from global used set
        old_diffs = get_diffs(old_row)
        current_used -= old_diffs
        
        new_row = None
        
        # Strategy: Try to modify the row to have a smaller max value.
        # We can try to "shrink" the row by replacing elements with smaller values.
        # Or we can try a random valid row with a constrained max.
        
        # Let's try to find a new row with max_val <= current_max (or slightly less)
        # that is valid and doesn't conflict with current_used.
        
        # To make it efficient, we don't try every possible row.
        # We try to perturb the existing row.
        # Pick a random position in the row (excluding 0) and change its value.
        
        # However, changing one value might break the increasing order or validity.
        # A safer mutation: Pick a random position `p` (1..k) and try to set `row[p]` to a new value `v`
        # such that `row[p-1] < v < row[p+1]` (if p < k).
        # Then check if the whole row is valid and disjoint from current_used.
        
        # To encourage reduction, we bias `v` to be smaller than `old_row[p]`.
        
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            attempts += 1
            p = rng.randint(1, k)
            
            min_v = old_row[p-1] + 1
            # We want to try smaller values first to reduce max
            # But we must maintain increasing order.
            # If p == k, max_v is old_row[k] (we want to reduce it)
            # If p < k, max_v is old_row[p+1] - 1
            
            if p < k:
                max_v = old_row[p+1] - 1
            else:
                max_v = old_row[p]
            
            if min_v > max_v:
                continue
                
            # Bias towards smaller values
            # Sample from [min_v, max_v]
            # To try to reduce max, if p==k, we prefer values < old_row[k]
            # If p < k, reducing v might not reduce the row max, but it opens space for later elements?
            # Actually, just randomizing is okay, but let's try a few specific strategies.
            
            # Strategy 1: Try a value smaller than current
            if old_row[p] > min_v:
                target_max = min(max_v, old_row[p] - 1)
                if target_max >= min_v:
                    v = rng.randint(min_v, target_max)
                else:
                    v = rng.randint(min_v, max_v)
            else:
                v = rng.randint(min_v, max_v)
                
            candidate = list(old_row)
            candidate[p] = v
            
            # Check increasing order
            valid_order = True
            for i in range(1, len(candidate)):
                if candidate[i] <= candidate[i-1]:
                    valid_order = False
                    break
            
            if not valid_order:
                continue
                
            # Check internal validity
            if not is_row_valid(candidate):
                continue
                
            # Check global validity
            cand_diffs = get_diffs(candidate)
            if cand_diffs & current_used:
                continue
            
            # Valid candidate found
            new_row = candidate
            break
            
        if new_row is None:
            # Revert: put old diffs back
            current_used |= old_diffs
            temperature *= cooling_rate
            continue
            
        # Update state
        cand_diffs = get_diffs(new_row)
        new_max = max(new_row)
        
        # Calculate delta in cost (max value)
        # Note: Changing one row might change the global max.
        # We track the global max by recalculating or keeping a count.
        # For simplicity and correctness, we recalculate global max if the mutated row was the one with the max,
        # or if the new row's max is larger than the previous global max.
        
        # Optimization: Keep track of which row has the max?
        # Simpler: Just recalculate max if the change is significant.
        # Since n_rows is small (7), recalculating max is O(n_rows * k) which is cheap.
        
        # Acceptance criteria
        # Cost is the global max.
        # We need to know the new global max.
        # Let's compute it.
        new_rows_temp = [list(r) for r in current_rows]
        new_rows_temp[row_idx] = new_row
        new_global_max = max(max(r) for r in new_rows_temp)
        
        delta = new_global_max - current_max
        
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            # Accept
            current_rows[row_idx] = new_row
            current_used |= cand_diffs
            current_max = new_global_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in current_rows]
        else:
            # Reject, but we already removed old_diffs. We must restore them.
            current_used |= old_diffs
            
        temperature *= cooling_rate
        
    return best_rows
