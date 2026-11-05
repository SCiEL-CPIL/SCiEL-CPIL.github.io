import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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
    def greedy_construct(seed, pressure_weight=1.0):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for _ in range(k):
                last = row[-1]
                valid_candidates = []
                
                # Find valid candidates
                for x in range(last + 1, 10000):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        max_new_diff = max(new_diffs)
                        score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                        valid_candidates.append((x, new_diffs, score))
                        if len(valid_candidates) >= 10:
                            break
                
                if not valid_candidates:
                    return None
                
                # Sort by score and pick from top few with some randomness
                valid_candidates.sort(key=lambda t: t[2])
                n_top = min(5, len(valid_candidates))
                top_candidates = valid_candidates[:n_top]
                
                # Weighted random selection: lower score = higher probability
                weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
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
                
            rows.append(row)
            used.update(local_used)
            
        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None
    
    # Try more seeds to find a good initial solution
    for seed in range(20):
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
        res = greedy_construct(0)
        if res:
            best_rows, current_used = res
            current_rows = best_rows
            best_max = get_max(best_rows)
        else:
            return []

    # Lower bound
    lb = max(k, n_rows * k * (k + 1) // 2)
    
    if best_max == lb:
        return best_rows

    # 2. Local Search with Rebuilds and Simulated Annealing
    rng = random.Random(42)
    current_max = best_max
    
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, pressure_weight=1.0, max_candidates=50):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        """
        row = [0]
        local_used = set()
        forbidden = forbidden_diffs | local_used
        
        for step in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates up to max_val_limit
            checked = 0
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    max_new_diff = max(new_diffs)
                    score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= max_candidates:
                        break
            
            if not valid_candidates:
                return None
            
            # Sort by score and pick from top few with some randomness
            valid_candidates.sort(key=lambda t: t[2])
            n_top = min(5, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]
            
            # Weighted random selection
            weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
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
            forbidden.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit):
        """
        Try to rebuild 2 rows simultaneously.
        """
        row_i_new = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_i_new is None:
            return None, None
        
        diffs_i = get_diffs(row_i_new)
        
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row(k, forbidden_j, rng, max_val_limit)
        if row_j_new is None:
            return None, None
        
        return row_i_new, row_j_new

    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    temp = initial_temp
    
    no_improve_count = 0
    total_moves = 0
    
    while not time_up():
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Check if we should switch to polish mode (last 15 seconds)
        polish_mode = remaining < 15.0
        
        if polish_mode:
            target_max = current_max - 1
            accept_neutral = False
            pressure_weight = 1.0
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.15 if temp > 0.1 else 0.05)
            pressure_weight = 0.8 + 0.2 * temp

        # Decide whether to do single or multi-row rebuild
        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)
        
        if use_multi:
            # Pick two distinct rows
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows
            
            old_row_i = current_rows[i]
            old_row_j = current_rows[j]
            old_diffs_i = get_diffs(old_row_i)
            old_diffs_j = get_diffs(old_row_j)
            
            # Remove both from used_diffs
            current_used.difference_update(old_diffs_i)
            current_used.difference_update(old_diffs_j)
            
            # Try to rebuild both
            new_row_i, new_row_j = rebuild_two_rows(
                k, current_used, rng, target_max if target_max > k else current_max
            )
            
            if new_row_i is None or new_row_j is None:
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)
            
            # Verify no conflicts
            if (new_diffs_i & current_used) or (new_diffs_j & current_used) or (new_diffs_i & new_diffs_j):
                current_used.update(old_diffs_i)
                current_used.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            # Calculate new global max
            current_rows[i] = new_row_i
            current_rows[j] = new_row_j
            new_global_max = get_max(current_rows)
            
            if new_global_max < current_max:
                current_used.update(new_diffs_i)
                current_used.update(new_diffs_j)
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)  # Reheat slightly
            else:
                if accept_neutral:
                    current_used.update(new_diffs_i)
                    current_used.update(new_diffs_j)
                    no_improve_count += 1
                else:
                    # Restore
                    current_rows[i] = old_row_i
                    current_rows[j] = old_row_j
                    current_used.update(old_diffs_i)
                    current_used.update(old_diffs_j)
                    no_improve_count += 1
            
            temp *= cooling_rate
            total_moves += 1
        else:
            # Single row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = current_rows[i]
            old_diffs = get_diffs(old_row)
            
            # Remove from used_diffs
            current_used.difference_update(old_diffs)
            
            new_row = None
            
            # Attempt 1: Improve (strictly smaller max for this row)
            if target_max > k:
                new_row = rebuild_row(k, current_used, rng, target_max, pressure_weight)
            
            # Attempt 2: Maintain/Reshuffle (if improve failed)
            if new_row is None:
                new_row = rebuild_row(k, current_used, rng, current_max, pressure_weight)
            
            if new_row is None:
                current_used.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs = get_diffs(new_row)
            
            if new_diffs & current_used:
                current_used.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            current_rows[i] = new_row
            new_global_max = get_max(current_rows)
            
            if new_global_max < current_max:
                current_used.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)
            else:
                if accept_neutral:
                    current_used.update(new_diffs)
                    no_improve_count += 1
                else:
                    current_rows[i] = old_row
                    current_used.update(old_diffs)
                    no_improve_count += 1
            
            temp *= cooling_rate
            total_moves += 1

    # Final polish pass: try to reduce max further with tighter constraints
    if not time_up():
        polish_deadline = min(deadline, time.time() + 12.0)
        while time.time() < polish_deadline:
            improved = False
            for i in range(n_rows):
                old_row = current_rows[i]
                old_diffs = get_diffs(old_row)
                current_used.difference_update(old_diffs)
                
                target_max = current_max - 1
                if target_max > k:
                    new_row = rebuild_row(k, current_used, rng, target_max, 1.0, max_candidates=30)
                    if new_row is not None:
                        new_diffs = get_diffs(new_row)
                        if not (new_diffs & current_used):
                            current_rows[i] = new_row
                            current_used.update(new_diffs)
                            current_max = get_max(current_rows)
                            best_rows = [r[:] for r in current_rows]
                            improved = True
                            break
                        else:
                            current_used.update(old_diffs)
                    else:
                        current_used.update(old_diffs)
                else:
                    current_used.update(old_diffs)
            
            if not improved:
                break

    return best_rows

# EVOLVE-BLOCK-END
