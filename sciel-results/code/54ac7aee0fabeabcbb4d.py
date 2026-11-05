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
    1. Fast randomized greedy to get a valid baseline.
    2. Local search with row-rebuild to tighten the maximum entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def time_up():
        return time.time() > deadline

    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    # Helper to compute all positive differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # 1. Multi-start Greedy with optimized candidate search
    def greedy_construct(seed, target_max=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for pos in range(k):
                last = row[-1]
                remaining = k - pos - 1
                # If we have a target max, constrain the search
                if target_max is not None:
                    max_x_allowed = target_max - remaining
                else:
                    max_x_allowed = 100000
                
                if last + 1 > max_x_allowed:
                    return None
                
                valid_candidates = []
                width = 8
                
                for x in range(last + 1, max_x_allowed + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None # Failed to construct
                
                # Selection strategy: pick randomly from top candidates
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
                
                row.append(x)
                local_used.update(new_diffs)
                
            rows.append(row)
            used.update(local_used)
            
        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None
    
    # Try seeds to find a good initial solution
    # First pass: no target max
    for seed in range(15):
        if time_up(): break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used = used
    
    # Second pass: try with target max = best_max to see if we can do better
    if best_rows is not None and best_max > lb:
        for seed in range(10):
            if time_up(): break
            target = best_max - 1
            if target <= k:
                break
            res = greedy_construct(seed, target)
            if res:
                rows, used = res
                m = get_max(rows)
                if m < best_max:
                    best_max = m
                    best_rows = [r[:] for r in rows]
                    current_rows = rows
                    current_used = used
    
    if best_rows is None:
        # Fallback: deterministic greedy
        res = greedy_construct(0)
        if res:
            best_rows, current_used = res
            current_rows = best_rows
            best_max = get_max(best_rows)
        else:
            # Should not happen for small k, n_rows
            return []

    if best_max == lb:
        return best_rows

    # 2. Local Search with Rebuilds
    # We maintain a set of used differences.
    # We repeatedly pick a row, remove its differences, and try to rebuild it with a smaller max.
    # If we succeed, we accept. If not, we try to rebuild with the same max (reshuffle) to escape local minima.
    
    rng = random.Random(42)
    current_max = best_max

    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, bias=0.0, temperature=1.0):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        bias: probability to pick a random valid candidate from the first 10, else smallest.
        temperature: controls exploration (higher = more random)
        """
        row = [0]
        local_used = set()
        
        for pos in range(k):
            last = row[-1]
            valid_candidates = []
            
            remaining = k - pos - 1
            max_x_allowed = max_val_limit - remaining
            
            if last + 1 > max_x_allowed:
                return None
            
            # Collect more candidates for better diversity when temperature is high
            width = 15 if temperature > 0.5 else 10
            
            for x in range(last + 1, max_x_allowed + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= width:
                        break
            
            if not valid_candidates:
                return None
            
            # Selection strategy with temperature
            if temperature > 0.5 and rng.random() < bias:
                # Weighted random selection: prefer smaller x but allow larger
                # Use exponential weighting
                weights = [1.0 / (1.0 + i * 0.1 * temperature) for i in range(len(valid_candidates))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                idx = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        idx = i
                        break
                x, new_diffs = valid_candidates[idx]
            elif bias > 0 and rng.random() < bias:
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
            else:
                x, new_diffs = valid_candidates[0]
            
            row.append(x)
            local_used.update(new_diffs)
            
        return row

    no_improve_count = 0
    max_no_improve = 50
    
    # Track which rows hold the current max for prioritization
    def get_max_rows(rows, current_max):
        """Return indices of rows that contain the current max value."""
        max_rows = []
        for i, row in enumerate(rows):
            if max(row) == current_max:
                max_rows.append(i)
        return max_rows
    
    # Temperature schedule: start high, decay over time
    base_temp = 1.0
    decay_rate = 0.995
    
    while not time_up():
        # Prioritize rows that hold the current maximum
        max_rows = get_max_rows(current_rows, current_max)
        if max_rows and rng.random() < 0.7:
            i = rng.choice(max_rows)
        else:
            i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove from used_diffs
        current_used.difference_update(old_diffs)
        
        new_row = None
        current_temp = base_temp * (decay_rate ** no_improve_count)
        
        # Attempt 1: If this row holds the max, try to reduce it
        if max(old_row) == current_max:
            target_max = current_max - 1
            if target_max > k: # Must be at least k
                new_row = rebuild_row(k, current_used, rng, target_max, bias=0.4, temperature=current_temp)
        
        # Attempt 2: Reshuffle (maintain max) to escape local minima
        if new_row is None:
            # Try to rebuild with the same max, but different structure
            # Higher temperature means more exploration
            temp_for_reshuffle = min(1.0, current_temp * 1.5)
            new_row = rebuild_row(k, current_used, rng, current_max, bias=0.7, temperature=temp_for_reshuffle)
        
        if new_row is None:
            # Failed to rebuild
            current_used.update(old_diffs)
            no_improve_count += 1
        else:
            # Success
            current_rows[i] = new_row
            new_diffs = get_diffs(new_row)
            current_used.update(new_diffs)
            
            new_global_max = get_max(current_rows)
            if new_global_max < current_max:
                current_max = new_global_max
                no_improve_count = 0
                # Reset temperature when we improve
                base_temp = 1.0
            else:
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Aggressive restart: try to rebuild multiple rows with fresh randomization
            no_improve_count = 0
            base_temp = 1.0  # Reset temperature
            
            # Try to improve by rebuilding all max-holding rows
            for _ in range(3):
                if time_up():
                    break
                max_rows = get_max_rows(current_rows, current_max)
                if not max_rows:
                    break
                i = rng.choice(max_rows)
                old_row = current_rows[i]
                old_diffs = get_diffs(old_row)
                current_used.difference_update(old_diffs)
                
                target_max = current_max - 1
                if target_max > k:
                    new_row = rebuild_row(k, current_used, rng, target_max, bias=0.3, temperature=1.0)
                    if new_row is not None:
                        current_rows[i] = new_row
                        current_used.update(get_diffs(new_row))
                        new_global_max = get_max(current_rows)
                        if new_global_max < current_max:
                            current_max = new_global_max
                    else:
                        current_used.update(old_diffs)
    
    # Validate
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
    
    # Keep track of the best solution found during local search
    if validate(current_rows, n_rows, k):
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in current_rows]
    
    # Final polish: try to improve with a tighter target
    if best_rows is not None and best_max > lb:
        for _ in range(5):
            if time_up():
                break
            target = best_max - 1
            if target <= k:
                break
            # Try to rebuild all rows with the tighter target
            new_rows = []
            new_used = set()
            success = True
            for r in best_rows:
                row_diffs = get_diffs(r)
                new_used.difference_update(row_diffs)
                new_row = rebuild_row(k, new_used, rng, target, bias=0.2, temperature=0.5)
                if new_row is None:
                    success = False
                    break
                new_rows.append(new_row)
                new_used.update(get_diffs(new_row))
            
            if success and validate(new_rows, n_rows, k):
                new_max = get_max(new_rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = new_rows
    
    if not validate(best_rows, n_rows, k):
        return [[0] * (k + 1)] * n_rows
    
    return best_rows
# EVOLVE-BLOCK-END
