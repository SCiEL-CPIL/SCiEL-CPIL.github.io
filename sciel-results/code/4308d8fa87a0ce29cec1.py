import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    k_plus_1 = k + 1

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

    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_solution(seed, target_max=None):
        """
        Generates a valid DTS using greedy.
        If target_max is provided, it tries to keep elements <= target_max.
        Returns None if it fails to find a valid row within the target.
        """
        random.seed(seed)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates
                # We want to keep values small, but if we have a target, we must stay under it
                # to potentially improve the global max.
                
                x = current_max + 1
                # Search window
                if target_max is not None:
                    search_limit = target_max
                else:
                    # Heuristic: search up to 2000 + row_idx * 100
                    search_limit = x + 2000 + row_idx * 100
                
                candidates = []
                while x <= search_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        candidates.append(x)
                        # Early exit if we have enough candidates?
                        # For greedy, we usually want the smallest, but randomized choice helps.
                        if len(candidates) > 20:
                            break
                    x += 1
                
                if not candidates:
                    if target_max is not None:
                        # Failed to find within target
                        return None
                    else:
                        # If no small candidates, search further without bound
                        x = current_max + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used) and len(new_diffs) == len(row):
                                candidates.append(x)
                                break
                            x += 1
                            if time.time() - start_time > time_limit - 5:
                                return None
                
                # Choose candidate
                # Sort candidates
                candidates.sort()
                # Take top 10 smallest
                top_cands = candidates[:10]
                
                # Probabilistic choice: 50% smallest, 30% next, 20% random from top 10
                # This introduces diversity in the generated solutions
                r = random.random()
                if r < 0.5:
                    chosen = top_cands[0]
                elif r < 0.8:
                    chosen = top_cands[1] if len(top_cands) > 1 else top_cands[0]
                else:
                    chosen = random.choice(top_cands)
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
            
            if time.time() - start_time > time_limit - 10:
                return None
        
        return rows

    def local_search(rows, max_time):
        """
        Local search to minimize the maximum element.
        Uses single-element moves and row-rebuild moves.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = compute_used(rows)
        
        # We will try to reduce the global maximum.
        # Strategy:
        # 1. Identify the element with the global maximum.
        # 2. Try to move it to a smaller value (single-element move).
        # 3. If that fails, try to rebuild the entire row containing it (row-rebuild move).
        
        while time.time() - start < max_time:
            # Find the element with the maximum value
            max_val = -1
            max_r = -1
            max_i = -1
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if rows[r][i] > max_val:
                        max_val = rows[r][i]
                        max_r = r
                        max_i = i
            
            if max_val == 0:
                break

            row = rows[max_r]
            current_val = row[max_i]
            
            # Attempt 1: Single-element move
            # Try to replace current_val with a smaller value v
            # Constraints:
            # 1. row[max_i-1] < v < row[max_i+1] (if neighbors exist)
            # 2. New diffs {v - a for a in row, a != v} must be unique and not in used (excluding old diffs of current_val)
            
            min_val = row[max_i - 1] + 1 if max_i > 0 else 1
            max_allowed = current_val - 1
            if max_i < len(row) - 1:
                max_allowed = min(max_allowed, row[max_i + 1] - 1)
            
            if max_allowed >= min_val:
                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                
                used_without_old = used - old_diffs
                
                found = False
                # Try candidates from min_val upwards, but limit the range for speed
                # If the range is huge, we might just try a few random ones or the smallest few
                search_range = min(max_allowed, min_val + 200)
                
                for v in range(min_val, search_range + 1):
                    new_diffs = set()
                    valid = True
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used_without_old or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                    
                    if valid:
                        # Apply move
                        row[max_i] = v
                        used = used_without_old | new_diffs
                        found = True
                        new_score = get_score(rows)
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [list(r) for r in rows]
                        break
                
                if found:
                    continue

            # Attempt 2: Row-rebuild move
            # If single-element move failed, rebuild the row max_r
            # We want to rebuild it such that its max is < current best_score
            if time.time() - start < max_time - 1:
                # Remove row max_r from used
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs.add(row[j] - row[i])
                
                used_without_row = used - row_diffs
                
                # Target max for the new row
                target_max = best_score - 1
                if target_max < k: # Must be at least k to fit k+1 elements starting at 0
                    target_max = k
                
                # Build new row for max_r
                new_row = [0]
                cur_max = 0
                success = True
                
                for _ in range(k):
                    x = cur_max + 1
                    found_x = False
                    # Search for x <= target_max
                    while x <= target_max:
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & used_without_row) and len(new_diffs) == len(new_row):
                            new_row.append(x)
                            used_without_row.update(new_diffs)
                            cur_max = x
                            found_x = True
                            break
                        x += 1
                        if time.time() - start > max_time - 0.1:
                            success = False
                            break
                    if not success:
                        break
                    if not found_x:
                        # If we can't find x <= target_max, this rebuild fails to improve
                        success = False
                        break
                
                if success:
                    rows[max_r] = new_row
                    used = compute_used(rows)
                    new_score = get_score(rows)
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                else:
                    # Restore state? We didn't modify rows yet, only used_without_row was a copy
                    # So no restoration needed for rows.
                    # But if we are stuck in a local minimum where we can't improve, we should break
                    # to avoid infinite loop.
                    break

        return best_rows

    # Main Logic
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Get a valid solution
    # Try a few seeds with greedy
    for seed in range(10):
        if time.time() - start_time > time_limit - 20:
            break
        rows = greedy_solution(seed)
        if rows and is_valid(rows):
            score = get_score(rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: Simple constructive without bound
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

    # Phase 2: Local Search
    # Allocate remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        final_rows = local_search(best_rows, max_time=remaining * 0.7)
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            best_score = final_score

    # Phase 3: If time remains, try restarting from best and perturbing?
    # Or just run local search again on the best found.
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        final_rows = local_search(best_rows, max_time=remaining * 0.9)
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            best_score = final_score

    return best_rows
