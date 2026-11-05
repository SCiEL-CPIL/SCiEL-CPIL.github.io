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

    # 1. Greedy Initial Solution
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

    # Rebuild a single row with pressure heuristic
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit):
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates
            # We limit the search depth to keep it fast, but ensure we check enough
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    # Score: prioritize smaller x, but also consider "future cost"
                    # A simple proxy for future cost is the number of new diffs introduced
                    # which is always len(row), so we just use x.
                    # However, to pack tighter, we might prefer x that doesn't leave large gaps?
                    # For now, standard greedy with random tie-breaking.
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 10:
                        break
            
            if not valid_candidates:
                return None

            # Pick randomly from the top few to allow exploration
            # But bias towards smaller x
            n_cand = len(valid_candidates)
            if n_cand == 1:
                chosen = valid_candidates[0]
            else:
                # Weighted selection: smaller x gets higher weight
                weights = [1.0 / (c[0] + 1) for c in valid_candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen = valid_candidates[0]
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = valid_candidates[i]
                        break

            x, new_diffs = chosen
            row.append(x)
            local_used.update(new_diffs)

        return row

    # Coordinated Rebuild of Two Rows
    # This is the key structural improvement: rebuilding two rows simultaneously
    # allows them to "trade" differences, potentially finding a valid configuration
    # where neither could be rebuilt individually within the current max.
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit):
        # Strategy: Build row A, then build row B with A's diffs added to forbidden set.
        # To make it symmetric and robust, we can try both orders or just pick one.
        # Let's build row A first.
        row_a = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_a is None:
            return None, None
        
        diffs_a = get_diffs(row_a)
        forbidden_b = forbidden_diffs | diffs_a
        
        # Now build row B
        row_b = rebuild_row(k, forbidden_b, rng, max_val_limit)
        if row_b is None:
            return None, None
            
        return row_a, row_b

    rng = random.Random(42)
    no_improve_count = 0
    iteration = 0
    sa_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01

    while time.time() < deadline:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Determine if we should use multi-row rebuilds
        # Use them more often when stuck or when remaining time allows
        use_multi = (n_rows >= 2) and (no_improve_count > 10 or rng.random() < 0.3)
        
        if use_multi:
            # Pick two distinct rows to rebuild
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows
            
            old_row_i = rows[i]
            old_row_j = rows[j]
            old_diffs_i = get_diffs(old_row_i)
            old_diffs_j = get_diffs(old_row_j)
            
            # Remove from used_diffs
            used_diffs.difference_update(old_diffs_i)
            used_diffs.difference_update(old_diffs_j)
            
            # Target max: try to improve
            target_max = current_max - 1
            if target_max <= k:
                target_max = current_max
                
            new_row_i, new_row_j = rebuild_two_rows(k, used_diffs, rng, target_max)
            
            if new_row_i is None or new_row_j is None:
                # Failed, try with current_max if we were targeting improvement
                if target_max < current_max:
                    new_row_i, new_row_j = rebuild_two_rows(k, used_diffs, rng, current_max)
            
            if new_row_i is None or new_row_j is None:
                # Restore
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
                continue
            
            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)
            
            # Verify no conflicts
            if (new_diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (new_diffs_i & new_diffs_j):
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
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
                sa_temp = min(sa_temp * 1.2, 1.0) # Reheat
            else:
                # Accept neutral moves with probability based on temperature
                if rng.random() < sa_temp * 0.1:
                    used_diffs.update(new_diffs_i)
                    used_diffs.update(new_diffs_j)
                    no_improve_count += 1
                else:
                    # Reject
                    rows[i] = old_row_i
                    rows[j] = old_row_j
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    no_improve_count += 1
                    sa_temp = max(sa_temp * cooling_rate, min_temp)
        
        else:
            # Single row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            used_diffs.difference_update(old_diffs)
            
            target_max = current_max - 1
            if target_max <= k:
                target_max = current_max
                
            new_row = rebuild_row(k, used_diffs, rng, target_max)
            
            if new_row is None:
                if target_max < current_max:
                    new_row = rebuild_row(k, used_diffs, rng, current_max)
            
            if new_row is None:
                used_diffs.update(old_diffs)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
                continue
            
            new_diffs = get_diffs(new_row)
            
            if new_diffs & used_diffs:
                used_diffs.update(old_diffs)
                no_improve_count += 1
                sa_temp = max(sa_temp * cooling_rate, min_temp)
                continue
            
            rows[i] = new_row
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                sa_temp = min(sa_temp * 1.2, 1.0)
            else:
                if rng.random() < sa_temp * 0.1:
                    used_diffs.update(new_diffs)
                    no_improve_count += 1
                else:
                    rows[i] = old_row
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
                    sa_temp = max(sa_temp * cooling_rate, min_temp)

        # Lower bound check
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    return best_rows

# EVOLVE-BLOCK-END
