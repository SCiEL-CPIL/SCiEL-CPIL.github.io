import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_diffs_set(rows):
        """Return set of all differences in rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct(randomize=False, seed=None):
        """
        Greedy construction.
        If randomize is True, picks from a window of valid candidates.
        """
        if seed is not None:
            random.seed(seed)
        
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                candidates = []
                # Look ahead a bit
                window = 20 if randomize else 1
                for x in range(current_max + 1, current_max + 1 + window):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                
                if not candidates:
                    # If stuck, expand window or fail
                    # For greedy, we just fail or expand indefinitely
                    # For random greedy, we might retry with larger window
                    expanded_candidates = []
                    for x in range(current_max + 1, current_max + 1 + 100):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            expanded_candidates.append(x)
                    if expanded_candidates:
                        candidates = expanded_candidates
                    else:
                        return None # Failed to construct

                if randomize and len(candidates) > 1:
                    # Bias towards smaller candidates
                    weights = [1.0 / (c - current_max + 1) for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break
                else:
                    next_x = candidates[0]
                
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            rows.append(row)
        return rows

    # Phase 1: Initialization
    # Try deterministic greedy first
    greedy_rows = greedy_construct(randomize=False)
    if greedy_rows and is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = get_scope(greedy_rows)
        if best_scope <= lower_bound:
            return best_rows

    # Try several randomized greedy starts
    for i in range(20):
        if time.time() > deadline - 5:
            break
        r = greedy_construct(randomize=True, seed=i)
        if r and is_valid(r):
            scope = get_scope(r)
            if scope < best_scope:
                best_scope = scope
                best_rows = [row[:] for row in r]
                if best_scope <= lower_bound:
                    return best_rows

    if best_rows is None:
        # Fallback: should not happen, but just in case
        return greedy_rows if greedy_rows else [[0]] * n_rows

    # Phase 2: Simulated Annealing
    current_rows = [row[:] for row in best_rows]
    current_scope = best_scope
    current_diffs = compute_diffs_set(current_rows)
    
    # Pre-compute row lengths
    row_len = k + 1
    
    temperature = 10.0
    cooling_rate = 0.999
    min_temp = 0.1
    
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline - 1:
        if temperature < min_temp:
            temperature = min_temp
        
        # Pick a random row and a random index to modify
        row_idx = random.randint(0, n_rows - 1)
        row = current_rows[row_idx]
        
        # We can only modify indices 1 to k (index 0 is always 0)
        if row_len <= 1:
            break
        idx = random.randint(1, row_len - 1)
        
        old_val = row[idx]
        
        # Determine valid range for new value
        min_val = row[idx - 1] + 1
        max_val = row[idx + 1] - 1 if idx < row_len - 1 else current_scope + 50 # Allow some expansion
        
        # If the range is empty, try a different move or skip
        if min_val > max_val:
            continue
            
        # Generate a candidate new value
        # We can try to reduce the value or move it randomly
        # To allow escaping local minima, we sample from the range
        # But we want to bias towards smaller values to reduce scope
        # Let's pick a random value in the range, but with a bias
        # Or simply pick a random integer in [min_val, max_val]
        
        # Optimization: Only consider values that might change the scope or are close to current
        # If old_val is the global max, we really want to reduce it.
        # Let's just sample uniformly from the valid range for simplicity, 
        # but limit the range to avoid huge jumps that are likely invalid
        # Actually, the validity check is O(k) for the row, which is small.
        
        # Let's try a few random candidates
        found_valid_move = False
        new_val = old_val
        
        # Number of candidates to try before giving up on this step
        num_tries = 10
        for _ in range(num_tries):
            if min_val == max_val:
                candidate = min_val
            else:
                # Bias towards smaller values
                # Use a triangular distribution or just uniform
                candidate = random.randint(min_val, max_val)
            
            if candidate == old_val:
                continue
                
            # Check validity of new_val
            # Remove old differences involving idx
            # old_diffs = {old_val - row[i] for i in range(idx)} | {row[j] - old_val for j in range(idx+1, row_len)}
            
            # New differences
            new_diffs = set()
            for i in range(idx):
                new_diffs.add(candidate - row[i])
            for j in range(idx + 1, row_len):
                new_diffs.add(row[j] - candidate)
                
            # Check if any new_diff is in current_diffs, 
            # BUT we must exclude the old_diffs that are being removed
            # So, effective_diffs = current_diffs - old_diffs
            # If new_diffs & effective_diffs is empty, it's valid.
            
            # To avoid computing old_diffs set every time (which is O(k)),
            # we can just check against current_diffs and if a conflict is found,
            # check if that conflict was one of the old_diffs.
            
            valid = True
            for d in new_diffs:
                if d in current_diffs:
                    # Check if d was an old difference involving idx
                    # d = old_val - row[i] or row[j] - old_val
                    is_old_diff = False
                    for i in range(idx):
                        if old_val - row[i] == d:
                            is_old_diff = True
                            break
                    if not is_old_diff:
                        for j in range(idx + 1, row_len):
                            if row[j] - old_val == d:
                                is_old_diff = True
                                break
                    if not is_old_diff:
                        valid = False
                        break
            
            if valid:
                new_val = candidate
                found_valid_move = True
                break
        
        if not found_valid_move:
            continue
            
        # Calculate new scope
        # If new_val > current_scope, new_scope = new_val
        # Else if old_val == current_scope and new_val < current_scope, we need to recompute max
        # Else new_scope = current_scope
        
        new_scope = current_scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope and new_val < current_scope:
            # Need to find new max
            # This is O(n_rows * k), which is acceptable occasionally
            max_val_found = 0
            for r in current_rows:
                # Temporarily apply change? No, just check
                # We can just take max of all rows, replacing row_idx's old_val with new_val
                pass # Optimization: maintain a frequency array of values? 
            # For now, just compute max
            # Since k is small, this is fast
            new_max = 0
            for ri, r in enumerate(current_rows):
                if ri == row_idx:
                    m = max(r)
                    if m == old_val:
                        # Find next max in this row
                        # Sort? No, just max of others
                        others = [x for i, x in enumerate(r) if i != idx]
                        m = max(others) if others else 0
                    new_max = max(new_max, m)
                else:
                    new_max = max(new_max, max(r))
            new_scope = new_max
            
        # Acceptance
        delta = new_scope - current_scope
        
        if delta < 0:
            # Better
            current_rows[row_idx][idx] = new_val
            # Update diffs set
            # Remove old diffs
            for i in range(idx):
                current_diffs.discard(old_val - current_rows[row_idx][i])
            for j in range(idx + 1, row_len):
                current_diffs.discard(current_rows[row_idx][j] - old_val)
            # Add new diffs
            for i in range(idx):
                current_diffs.add(new_val - current_rows[row_idx][i])
            for j in range(idx + 1, row_len):
                current_diffs.add(current_rows[row_idx][j] - new_val)
                
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                if best_scope <= lower_bound:
                    return best_rows
            no_improve_count = 0
        elif delta == 0:
            # Equal, accept with some probability to explore
            if random.random() < 0.5:
                current_rows[row_idx][idx] = new_val
                for i in range(idx):
                    current_diffs.discard(old_val - current_rows[row_idx][i])
                for j in range(idx + 1, row_len):
                    current_diffs.discard(current_rows[row_idx][j] - old_val)
                for i in range(idx):
                    current_diffs.add(new_val - current_rows[row_idx][i])
                for j in range(idx + 1, row_len):
                    current_diffs.add(current_rows[row_idx][j] - new_val)
                no_improve_count += 1
        else:
            # Worse, accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / temperature):
                current_rows[row_idx][idx] = new_val
                for i in range(idx):
                    current_diffs.discard(old_val - current_rows[row_idx][i])
                for j in range(idx + 1, row_len):
                    current_diffs.discard(current_rows[row_idx][j] - old_val)
                for i in range(idx):
                    current_diffs.add(new_val - current_rows[row_idx][i])
                for j in range(idx + 1, row_len):
                    current_diffs.add(current_rows[row_idx][j] - new_val)
                current_scope = new_scope
                no_improve_count += 1
            else:
                no_improve_count += 1

        temperature *= cooling_rate
        steps += 1
        
        # If we haven't improved in a long time, restart from best known
        if no_improve_count > 500:
            current_rows = [r[:] for r in best_rows]
            current_scope = best_scope
            current_diffs = compute_diffs_set(current_rows)
            temperature = 5.0
            no_improve_count = 0

    return best_rows
