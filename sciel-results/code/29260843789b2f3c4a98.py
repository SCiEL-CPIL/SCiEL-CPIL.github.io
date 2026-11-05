import random
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Build a solid initial solution using a fast greedy algorithm.
    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # Helper to compute max of rows
    def get_max(rows):
        return max(max(r) for r in rows)

    # Helper to compute diffs of a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # 2. Improved rebuild with difference-pressure candidate selection
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, pressure_weight=0.7):
        """
        Builds a row using a hybrid strategy:
        - Collect valid candidates
        - Score each candidate by a combination of its value and the max new difference it introduces
        - Prefer candidates with low score (small value, low max new diff)
        """
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates up to max_val_limit
            # Limit search to avoid excessive time; we'll collect up to a reasonable number
            max_candidates_to_check = 80
            checked = 0
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    # Score: combination of x and max new diff
                    max_new_diff = max(new_diffs)
                    score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= max_candidates_to_check:
                        break
            
            if not valid_candidates:
                return None
            
            # Sort by score and pick from top few with some randomness
            valid_candidates.sort(key=lambda t: t[2])
            
            # Pick from top min(8, len) candidates with weighted probability favoring lower scores
            n_top = min(8, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]
            
            # Weighted random selection: lower score = higher probability
            weights = [1.0 / (1.0 + c[2] * 0.05) for c in top_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = top_candidates[0]
            for c, w in zip(top_candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break
            
            x, new_diffs, _ = chosen
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    # 3. Multi-row rebuild: try to rebuild 2 rows simultaneously
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit, row_i, row_j):
        """
        Try to rebuild rows i and j simultaneously.
        Returns (new_row_i, new_row_j) or (None, None) if failed.
        """
        # First, build row i with forbidden_diffs
        row_i_new = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_i_new is None:
            return None, None
        
        diffs_i = get_diffs(row_i_new)
        
        # Now build row j with forbidden_diffs + diffs_i
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row(k, forbidden_j, rng, max_val_limit)
        if row_j_new is None:
            return None, None
        
        return row_i_new, row_j_new

    # Main Loop with simulated annealing
    rng = random.Random(12345)
    
    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.997
    min_temp = 0.005
    temp = initial_temp
    
    no_improve_count = 0
    total_moves = 0
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Check if we should switch to polish mode (last 15 seconds)
        polish_mode = remaining < 15.0
        
        if polish_mode:
            # In polish mode, only accept improvements and use tighter limits
            target_max = current_max - 1
            accept_neutral = False
            max_candidates = 30
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.20 if temp > 0.1 else 0.08)
            max_candidates = 50

        # Decide whether to do single or multi-row rebuild
        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)
        
        if use_multi:
            # Pick two distinct rows
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows
            
            old_row_i = rows[i]
            old_row_j = rows[j]
            old_diffs_i = get_diffs(old_row_i)
            old_diffs_j = get_diffs(old_row_j)
            
            # Remove both from used_diffs
            used_diffs.difference_update(old_diffs_i)
            used_diffs.difference_update(old_diffs_j)
            
            # Try to rebuild both
            new_row_i, new_row_j = rebuild_two_rows(
                k, used_diffs, rng, target_max if target_max > k else current_max, i, j
            )
            
            if new_row_i is None or new_row_j is None:
                # Failed, restore
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)
            
            # Verify no conflicts
            if (new_diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (new_diffs_i & new_diffs_j):
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            # Calculate new global max
            rows[i] = new_row_i
            rows[j] = new_row_j
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs_i)
                used_diffs.update(new_diffs_j)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)  # Reheat slightly
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs_i)
                    used_diffs.update(new_diffs_j)
                else:
                    # Reject: restore old rows
                    rows[i] = old_row_i
                    rows[j] = old_row_j
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
        else:
            # Single row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            # Remove from used_diffs
            used_diffs.difference_update(old_diffs)
            
            # Try to rebuild
            new_row = None
            # First, try to improve (find row with max < current_max)
            target_max = current_max - 1
            if target_max > k: # Must be at least k
                new_row = rebuild_row(k, used_diffs, rng, target_max)
            
            # If failed, attempt to maintain (neutral move)
            if new_row is None:
                new_row = rebuild_row(k, used_diffs, rng, current_max)
            
            if new_row is None:
                # Rebuild failed completely. Restore old row.
                used_diffs.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs = get_diffs(new_row)
            
            # Verify no conflicts (should be true by construction)
            if new_diffs & used_diffs:
                # Bug in rebuild? Restore.
                used_diffs.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
                
            # Calculate new global max
            # Temporarily replace row i
            rows[i] = new_row
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                # Improvement!
                used_diffs.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)  # Reheat slightly
            else:
                # No improvement (or worse, though we capped at current_max so it shouldn't be worse)
                # Accept with some probability? 
                if accept_neutral:
                    used_diffs.update(new_diffs)
                    # current_max stays same
                    no_improve_count += 1
                else:
                    # Reject: restore old row
                    rows[i] = old_row
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1

        # If no_improve_count is very high, we might be in a local minimum.
        # We could try a "big jump": rebuild multiple rows at once?
        # Or just accept that we've hit a local optimum for this strategy.
        # Given the 120s limit, we just keep churning.
        
        # Optional: If current_max is close to theoretical lower bound, we might be done.
        # Lower bound: n_rows * k(k+1)/2
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    # Final polish pass: try to reduce max further with aggressive single-row rebuilds
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        polish_rng = random.Random(99999)
        polish_deadline = time.time() + max(0, remaining_time - 1.0)
        
        # Sort rows by their max entry in descending order (largest first)
        for _ in range(3):  # Multiple passes
            if time.time() >= polish_deadline:
                break
            # Get rows sorted by max entry
            row_indices = sorted(range(n_rows), key=lambda i: max(rows[i]), reverse=True)
            for idx in row_indices:
                if time.time() >= polish_deadline:
                    break
                old_row = rows[idx]
                old_max = max(old_row)
                if old_max < current_max:
                    continue
                old_diffs = get_diffs(old_row)
                used_diffs.difference_update(old_diffs)
                
                # Try to rebuild this row with a tighter limit
                new_row = rebuild_row(k, used_diffs, polish_rng, current_max - 1)
                
                if new_row is not None:
                    new_diffs = get_diffs(new_row)
                    if not (new_diffs & used_diffs):
                        rows[idx] = new_row
                        used_diffs.update(new_diffs)
                        new_global_max = get_max(rows)
                        if new_global_max < current_max:
                            current_max = new_global_max
                            best_rows = [r[:] for r in rows]
                    else:
                        used_diffs.update(old_diffs)
                else:
                    used_diffs.update(old_diffs)
        
        # One more pass with even tighter limits
        for _ in range(2):
            if time.time() >= polish_deadline:
                break
            row_indices = sorted(range(n_rows), key=lambda i: max(rows[i]), reverse=True)
            for idx in row_indices:
                if time.time() >= polish_deadline:
                    break
                old_row = rows[idx]
                old_max = max(old_row)
                if old_max < current_max:
                    continue
                old_diffs = get_diffs(old_row)
                used_diffs.difference_update(old_diffs)
                
                # Try with current_max - 2
                new_row = rebuild_row(k, used_diffs, polish_rng, current_max - 2)
                
                if new_row is not None:
                    new_diffs = get_diffs(new_row)
                    if not (new_diffs & used_diffs):
                        rows[idx] = new_row
                        used_diffs.update(new_diffs)
                        new_global_max = get_max(rows)
                        if new_global_max < current_max:
                            current_max = new_global_max
                            best_rows = [r[:] for r in rows]
                    else:
                        used_diffs.update(old_diffs)
                else:
                    used_diffs.update(old_diffs)

    return best_rows

# EVOLVE-BLOCK-END
