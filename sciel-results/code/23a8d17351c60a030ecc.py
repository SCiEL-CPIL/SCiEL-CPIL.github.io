import random
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    def time_up():
        return time.time() > deadline

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

    # 1. Multi-start Greedy with Dynamic Pressure Ordering
    # Strategy: 
    # - Try multiple random orderings of rows? No, rows are indistinguishable in objective.
    # - Instead, use randomized greedy construction.
    # - "Difference Pressure": When building a row, prefer candidates that minimize the "cost" of adding new differences.
    #   Cost can be defined as the max of the new differences introduced, or the sum, or just the new max.
    #   The current greedy picks the smallest x that doesn't conflict. This naturally minimizes the new max difference.
    #   Let's introduce a "pressure" parameter that biases towards keeping the row compact.
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        # Randomize the order in which we build rows? 
        # Since rows are symmetric, the order doesn't matter for the final set validity, 
        # but it might affect the "greedy trap" we fall into.
        # However, usually building them in order is fine. 
        # Let's try shuffling the "priority" of which difference to use? No, we build row by row.
        
        # Let's try a randomized greedy where we pick from the top M smallest valid candidates.
        # M can vary.
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for _ in range(k):
                last = row[-1]
                valid_candidates = []
                
                # Find valid candidates
                # We scan x from last+1 upwards.
                # We collect a few smallest valid ones to allow randomization.
                # If we just pick the smallest, it's deterministic greedy.
                # Let's pick the smallest valid x, but with a small probability pick a slightly larger one?
                # Or better: pick from the first `choice_width` valid candidates.
                
                # To implement "dynamic pressure", we can adjust `choice_width` based on how tight the constraints are.
                # If we are close to the limit, we should be more greedy (width=1).
                # If we are loose, we can explore (width>1).
                
                # For now, let's use a fixed small width for randomization.
                width = 3
                
                for x in range(last + 1, 10000): # Upper bound for search in greedy phase
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None # Failed to construct
                
                # Select candidate
                # Heuristic: Pick the one that minimizes the new max difference?
                # Since x increases, new_diffs = {x - a}. The max new diff is x - row[0] = x.
                # So picking smallest x minimizes the immediate max.
                # Randomization: Pick randomly from the top `width` smallest.
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
    
    # Try a few seeds to find a good initial solution
    for seed in range(10):
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used = used
    
    if best_rows is None:
        # Fallback to deterministic greedy if random fails (unlikely)
        res = greedy_construct(0)
        if res:
            best_rows, current_used = res
            current_rows = best_rows
            best_max = get_max(best_rows)
        else:
            return [] # Should not happen
    
    # Lower bound
    lb = max(k, n_rows * k * (k + 1) // 2)
    
    if best_max == lb:
        return best_rows

    # 2. Local Search with Rebuilds
    rng = random.Random(42)
    current_max = best_max
    
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, bias=0.0):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        bias: probability to pick a random valid candidate from the first 10, else smallest.
        """
        row = [0]
        local_used = set()
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 10:
                        break
            
            if not valid_candidates:
                return None
            
            # Selection strategy
            if bias > 0 and rng.random() < bias:
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
            else:
                x, new_diffs = valid_candidates[0]
            
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    # Main Loop
    no_improve_count = 0
    
    while not time_up():
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove from used_diffs
        current_used.difference_update(old_diffs)
        
        new_row = None
        
        # Attempt 1: Improve (strictly smaller max for this row)
        target_max = current_max - 1
        if target_max > k:
            new_row = rebuild_row(k, current_used, rng, target_max, bias=0.5)
            
        # Attempt 2: Maintain/Reshuffle (if improve failed)
        if new_row is None:
            new_row = rebuild_row(k, current_used, rng, current_max, bias=0.8)
        
        if new_row is None:
            current_used.update(old_diffs)
            no_improve_count += 1
            continue
        
        new_diffs = get_diffs(new_row)
        
        if new_diffs & current_used:
            current_used.update(old_diffs)
            no_improve_count += 1
            continue
            
        current_rows[i] = new_row
        new_global_max = get_max(current_rows)
        
        if new_global_max < current_max:
            current_used.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in current_rows]
            no_improve_count = 0
            
            if current_max == lb:
                break
        else:
            if rng.random() < 0.2:
                current_used.update(new_diffs)
                no_improve_count += 1
            else:
                current_rows[i] = old_row
                current_used.update(old_diffs)
                no_improve_count += 1
        
    return best_rows

# EVOLVE-BLOCK-END
