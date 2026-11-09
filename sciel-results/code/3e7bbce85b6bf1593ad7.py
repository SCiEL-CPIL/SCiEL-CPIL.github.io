import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
        
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_solution(seed=0, fixed_used=None, target_row_idx=None):
        """
        Generate a solution using randomized greedy.
        If fixed_used is provided, it treats those differences as already taken
        and only generates the row at target_row_idx (or all if None, but usually
        used for rebuilds of a single row).
        """
        random.seed(seed)
        used = set()
        if fixed_used is not None:
            used.update(fixed_used)
            
        rows = []
        
        # If we are rebuilding a specific row, we need placeholders for others?
        # Actually, the caller passes the full rows and we just replace one.
        # This function is primarily for the initial full generation.
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates
                candidates = []
                x = current_max + 1
                # Adaptive limit: wider search for better diversity
                limit = x + 1000 + row_idx * 50
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 50:
                            break
                    x += 1
                
                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                
                # Weighted random selection: prefer smaller values but with diversity
                if len(candidates) <= 3:
                    chosen = random.choice(candidates)
                else:
                    # Use weights that favor smaller but allow exploration
                    # 1/(1+c*alpha)
                    alpha = 0.005
                    weights = [1.0 / (1.0 + c * alpha) for c in candidates[:20]]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates[:20], weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = c
                            break
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows

    def rebuild_row(rows, row_idx, seed, max_time=2.0):
        """
        Attempts to rebuild a specific row to reduce the global max.
        Returns the new rows if an improvement is found, else None.
        """
        start = time.time()
        other_rows = [r for i, r in enumerate(rows) if i != row_idx]
        
        # Compute differences used by other rows
        used_by_others = set()
        for r in other_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_by_others.add(r[j] - r[i])
        
        # We want to construct a new row for row_idx using only diffs NOT in used_by_others
        # and ensuring internal consistency.
        # We use a greedy approach similar to greedy_solution but for a single row.
        
        random.seed(seed)
        
        # We can try multiple random seeds for the rebuild
        best_new_row = None
        best_new_max = max(rows[row_idx])
        
        # We also want to see if we can beat the current global max
        current_global_max = get_score(rows)
        
        for attempt in range(10):
            if time.time() - start > max_time:
                break
            
            row = [0]
            current_max = 0
            valid = True
            # We need to ensure diffs within this row are unique and not in used_by_others
            # Note: diffs within this row are automatically unique if the row is strictly increasing?
            # No, differences between different pairs (i,j) can collide.
            # e.g. row [0, 2, 4] has diffs 2, 4, 2. Collision!
            # So we must track diffs within the new row too.
            
            new_row_used = set()
            
            while len(row) < k_plus_1:
                x = current_max + 1
                found = False
                # Search for a valid x
                # We want to keep x small to minimize the max
                # But we must avoid collisions with used_by_others AND new_row_used
                limit = x + 500
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_by_others) and not (new_diffs & new_row_used):
                        # Check if new_diffs have internal collisions?
                        # new_diffs is a set, so if len(new_diffs) == len(row), they are unique.
                        if len(new_diffs) == len(row):
                            row.append(x)
                            new_row_used.update(new_diffs)
                            current_max = x
                            found = True
                            break
                    x += 1
                if not found:
                    # If we can't find a small one, maybe we can't build this row this way
                    # Or we need to go further out. For simplicity, if we fail, abandon this attempt.
                    valid = False
                    break
            
            if valid:
                row_max = max(row)
                if row_max < best_new_max:
                    best_new_max = row_max
                    best_new_row = row
        
        if best_new_row is not None:
            # Construct new full rows
            new_rows = [list(r) for r in rows]
            new_rows[row_idx] = best_new_row
            # The score improvement is determined by the global max
            # If best_new_max < current_global_max, it's a definite improvement for this row's contribution
            # But another row might have a larger max.
            # However, since we only changed this row, if its max was the global max, we reduced it.
            # If it wasn't, the global max didn't change (unless we increased it, which we didn't).
            # So this is always a non-regressive move, and potentially improving.
            return new_rows
            
        return None

    def local_search(rows, max_time=5.0):
        """
        Hybrid local search: Shrink elements, Rebuild rows, Swap elements.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = compute_used(rows)
        
        improved = True
        stall_count = 0
        seed_counter = 0
        
        while improved and time.time() - start < max_time:
            improved = False
            stall_count += 1
            
            # Strategy 1: Shrink largest elements
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            
            found_improvement = False
            for val, max_r, max_i in elements[:5]:
                if time.time() - start > max_time - 0.1:
                    break
                
                row = rows[max_r]
                current_val = row[max_i]
                if current_val <= 0:
                    continue
                
                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)
                
                if max_allowed < min_val:
                    continue
                
                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                used -= old_diffs
                
                new_val_found = None
                # Try smallest valid replacement
                candidates_to_try = list(range(min_val, min(current_val, min_val + 100)))
                
                for v in candidates_to_try:
                    valid = True
                    new_diffs = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid and len(new_diffs) == len(row) - 1:
                        new_val_found = v
                        break
                
                if new_val_found is not None:
                    row[max_i] = new_val_found
                    used |= new_diffs
                    found_improvement = True
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    break
                else:
                    used |= old_diffs
            
            if found_improvement:
                improved = True
                continue
            
            # Strategy 2: Rebuild the row with the current global maximum
            # Identify row with max
            max_val = -1
            max_r = -1
            for r in range(n_rows):
                if max(rows[r]) > max_val:
                    max_val = max(rows[r])
                    max_r = r
            
            if max_r != -1 and time.time() - start < max_time - 0.5:
                seed_counter += 1
                new_rows = rebuild_row(rows, max_r, seed_counter, max_time=1.0)
                if new_rows:
                    score = get_score(new_rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in new_rows]
                        rows = new_rows
                        used = compute_used(rows)
                        improved = True
                        continue

            # Strategy 3: Swap (less frequent, as rebuild is stronger)
            if stall_count < 3 and time.time() - start < max_time - 0.5:
                swap_done = False
                for r1 in range(n_rows):
                    for r2 in range(r1 + 1, n_rows):
                        if time.time() - start > max_time - 0.2:
                            break
                        for i1 in range(1, len(rows[r1])):
                            for i2 in range(1, len(rows[r2])):
                                v1 = rows[r1][i1]
                                v2 = rows[r2][i2]
                                if v1 == v2:
                                    continue
                                
                                if i1 > 0 and v2 <= rows[r1][i1-1]:
                                    continue
                                if i1 < len(rows[r1])-1 and v2 >= rows[r1][i1+1]:
                                    continue
                                if i2 > 0 and v1 <= rows[r2][i2-1]:
                                    continue
                                if i2 < len(rows[r2])-1 and v1 >= rows[r2][i2+1]:
                                    continue
                                    
                                old_diffs1 = set()
                                for a in rows[r1]:
                                    if a != v1:
                                        old_diffs1.add(abs(a - v1))
                                old_diffs2 = set()
                                for a in rows[r2]:
                                    if a != v2:
                                        old_diffs2.add(abs(a - v2))
                                used -= (old_diffs1 | old_diffs2)
                                
                                new_diffs1 = set()
                                valid = True
                                for a in rows[r1]:
                                    if a != v1:
                                        d = abs(a - v2)
                                        if d in used:
                                            valid = False
                                            break
                                        new_diffs1.add(d)
                                if valid:
                                    new_diffs2 = set()
                                    for a in rows[r2]:
                                        if a != v2:
                                            d = abs(a - v1)
                                            if d in used:
                                                valid = False
                                                break
                                            new_diffs2.add(d)
                                
                                if valid:
                                    rows[r1][i1] = v2
                                    rows[r2][i2] = v1
                                    used |= (new_diffs1 | new_diffs2)
                                    swap_done = True
                                    found_improvement = True
                                    score = get_score(rows)
                                    if score < best_score:
                                        best_score = score
                                        best_rows = [list(r) for r in rows]
                                    break
                                else:
                                    used |= (old_diffs1 | old_diffs2)
                            if swap_done:
                                break
                        if swap_done:
                            break
                    if swap_done:
                        break
                
                if swap_done:
                    improved = True
                    continue

            if not found_improvement:
                improved = False
                stall_count += 1
                if stall_count > 5:
                    break
        
        return best_rows

    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds
    num_seeds = 100 # Reduced count to allow more time per seed for local search
    
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 10:
            break
        
        # Allocate time for local search
        # Early seeds: less LS. Later seeds: more LS.
        if seed < 20:
            ls_time = min(2.0, remaining * 0.1)
        elif seed < 50:
            ls_time = min(5.0, remaining * 0.15)
        else:
            ls_time = min(8.0, remaining * 0.2)
            
        if ls_time < 0.5:
            break
        
        rows = greedy_solution(seed)
        if rows is None:
            continue
        
        # Apply local search
        rows = local_search(rows, max_time=ls_time)
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            if score <= 112:
                break
        
        # Intensive polish if we are close to the bound
        if best_score <= 130 and remaining > 20 and seed > 30:
            final_rows = local_search(best_rows, max_time=min(remaining * 0.3, 15.0))
            final_score = get_score(final_rows)
            if final_score < best_score:
                best_score = final_score
                best_rows = [list(r) for r in final_rows]

    if best_rows is None:
        # Fallback
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
    
    # Final polish with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        final_rows = local_search(best_rows, max_time=min(remaining * 0.8, 10.0))
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            
    return best_rows
