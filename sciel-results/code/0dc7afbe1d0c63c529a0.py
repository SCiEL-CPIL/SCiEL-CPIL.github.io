import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    k_plus_1 = k + 1
    
    # Helper to create a bitmask from a set of differences
    def set_to_mask(diff_set):
        mask = 0
        for d in diff_set:
            if d > 0:
                mask |= (1 << d)
        return mask

    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True

    def greedy_solution(seed):
        random.seed(seed)
        used_mask = 0
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            # Upper bound for greedy search: start loose, tighten if needed
            # A safe upper bound is roughly k * k * 2 or similar, but we just go up
            # until we find enough candidates.
            search_limit = 100 + row_idx * 10 + k * 10
            
            while len(row) < k_plus_1:
                found = False
                # Try to find a candidate
                # We prefer smaller candidates to keep scope low
                candidates = []
                x = current_max + 1
                while x <= search_limit:
                    # Calculate new diffs
                    new_diffs = []
                    valid = True
                    mask = 0
                    for a in row:
                        d = x - a
                        if d <= 0: continue
                        # Check if d is already used in used_mask
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs.append(d)
                        mask |= (1 << d)
                    
                    if valid and len(new_diffs) == len(row):
                        candidates.append((x, mask))
                        if len(candidates) > 20:
                            break
                    x += 1
                
                if not candidates:
                    # If no candidates found in initial limit, expand limit
                    # This should rarely happen for reasonable k
                    search_limit *= 2
                    if search_limit > 10000:
                        return None
                    continue
                
                # Pick a candidate
                # Randomly pick from the first few valid candidates to introduce diversity
                # but bias towards smaller x
                if len(candidates) > 5:
                    choice_idx = random.randint(0, min(5, len(candidates)-1))
                else:
                    choice_idx = 0
                    
                chosen_x, chosen_mask = candidates[choice_idx]
                row.append(chosen_x)
                used_mask |= chosen_mask
                current_max = chosen_x
                
            rows.append(row)
            
        return rows, used_mask

    # Main Logic
    best_rows = None
    best_score = float('inf')
    best_used_mask = 0

    # Phase 1: Get valid solutions using Greedy
    # Try a few seeds to find a good starting point
    for seed in range(10):
        if time.time() - start_time > time_limit - 30:
            break
        res = greedy_solution(seed)
        if res:
            rows, mask = res
            if is_valid(rows):
                score = get_score(rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]
                    best_used_mask = mask

    if best_rows is None:
        # Fallback: Simple sequential construction (might be large but valid)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Phase 2: Local Search with Bitmask Optimization
    # We will run a local search that tries to reduce the max element.
    # It allows moving any element to any valid position.
    
    current_rows = [list(r) for r in best_rows]
    current_mask = best_used_mask
    current_score = best_score
    
    # Precompute row diffs for quick removal? 
    # It's easier to just recalculate diffs for a specific move.
    # A move affects only one row.
    
    end_time = time_limit - 5 # Leave 5s for final check and return
    
    # Simulated Annealing / Random Walk parameters
    temperature = 10.0
    cooling_rate = 0.99
    
    no_improve_count = 0
    max_no_improve = 2000 # Restart or cool down if stuck
    
    while time.time() < end_time:
        # Pick a random row and a random element (excluding the first, which is 0)
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k) # Indices 1 to k
        
        row = current_rows[r_idx]
        old_val = row[j_idx]
        
        # Determine valid range for new value
        # Must be strictly increasing
        low_bound = row[j_idx - 1] + 1
        if j_idx < k:
            high_bound = row[j_idx + 1] - 1
        else:
            # If it's the last element, we can potentially increase it, 
            # but we want to minimize max, so we usually try to keep it same or smaller.
            # However, to escape local minima, we might need to increase it temporarily.
            # Let's allow a small increase.
            high_bound = old_val + 10 
        
        if low_bound > high_bound:
            continue
            
        # If we are trying to minimize max, we should bias towards smaller values.
        # But for exploration, we pick randomly.
        # Let's pick a random value in the range.
        new_val = random.randint(low_bound, high_bound)
        
        if new_val == old_val:
            continue
            
        # Calculate diffs to remove
        diffs_to_remove = []
        for i in range(k_plus_1):
            if i == j_idx: continue
            d = abs(old_val - row[i])
            if d > 0:
                diffs_to_remove.append(d)
        
        # Calculate diffs to add
        diffs_to_add = []
        valid = True
        for i in range(k_plus_1):
            if i == j_idx: continue
            d = abs(new_val - row[i])
            if d <= 0:
                valid = False
                break
            diffs_to_add.append(d)
        
        if not valid or len(diffs_to_add) != k:
            continue
            
        # Check for conflicts using bitmask
        # A conflict exists if a diff in diffs_to_add is in current_mask 
        # AND that diff is NOT in diffs_to_remove (i.e., it's not being freed up by this move)
        
        # Optimization: 
        # 1. Create mask of diffs_to_remove
        rem_mask = 0
        for d in diffs_to_remove:
            rem_mask |= (1 << d)
            
        # 2. Check each diff in diffs_to_add
        conflict = False
        add_mask = 0
        for d in diffs_to_add:
            add_mask |= (1 << d)
            # If d is in current_mask, it's a conflict UNLESS it's in rem_mask
            # (if it's in rem_mask, it will be removed, so it's available)
            if (current_mask >> d) & 1:
                if not ((rem_mask >> d) & 1):
                    conflict = True
                    break
        
        if conflict:
            continue
            
        # Move is valid. Apply it.
        # Update mask
        new_mask = (current_mask & ~rem_mask) | add_mask
        
        # Update row
        row[j_idx] = new_val
        
        # Calculate new score
        new_score = get_score(current_rows)
        
        # Acceptance Criterion
        # We want to minimize new_score.
        # If new_score < current_score: Accept always.
        # If new_score == current_score: Accept with prob 0.5 (or always, to explore).
        # If new_score > current_score: Accept with prob exp(-delta / temperature).
        
        delta = new_score - current_score
        
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.5
        else:
            # Only accept worse moves if temperature is high enough
            if temperature > 1.0:
                accept = random.random() < math.exp(-delta / temperature)
            else:
                accept = False
                
        if accept:
            current_mask = new_mask
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_rows = [list(r) for r in current_rows]
                best_used_mask = new_mask
        else:
            # Revert move
            row[j_idx] = old_val
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Cool down
                temperature *= cooling_rate
                no_improve_count = 0
                if temperature < 0.1:
                    # Reset to try a different local region? 
                    # Or just stop if stuck for too long.
                    # Let's try to restart the local search from the best solution found so far
                    current_rows = [list(r) for r in best_rows]
                    current_mask = best_used_mask
                    current_score = best_score
                    temperature = 10.0
                    no_improve_count = 0

    # Final Verification
    if is_valid(best_rows):
        return best_rows
    else:
        # Should not happen, but fallback to the last known valid greedy solution if it exists
        # In this implementation, best_rows is only updated if valid.
        return best_rows
