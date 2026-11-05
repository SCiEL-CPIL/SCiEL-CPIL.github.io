import time
import random

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    start_time = time.time()
    deadline = 110.0  # Safe margin
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def is_valid_move(row_idx, new_row, old_row, current_used):
        new_diffs = get_row_diffs(new_row)
        old_diffs = get_row_diffs(old_row)
        # Check if any new diff is in current_used but NOT in old_diffs
        for d in new_diffs:
            if d in current_used:
                if d not in old_diffs:
                    return False
        return True

    max_iters = 200000
    iters = 0
    stuck = 0
    
    while time.time() - start_time < deadline and iters < max_iters:
        iters += 1
        
        # Find current max
        curr_max = -1
        max_r_idx = -1
        for i, r in enumerate(rows):
            if r[-1] > curr_max:
                curr_max = r[-1]
                max_r_idx = i
                
        if curr_max < best_max:
            best_max = curr_max
            best_rows = [r[:] for r in rows]
            stuck = 0
        else:
            stuck += 1
            
        # If stuck, do random perturbation
        if stuck > 200:
            # Random valid move on any row
            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            # Pick random pos
            pos = random.randint(1, k)
            old_val = old_row[pos]
            
            # Try small shifts
            for _ in range(10):
                delta = random.randint(-5, 5)
                if delta == 0: continue
                new_val = old_val + delta
                if new_val <= 0: continue
                if pos > 0 and new_val <= old_row[pos-1]: continue
                if pos < k and new_val >= old_row[pos+1]: continue
                
                new_row = old_row[:]
                new_row[pos] = new_val
                
                if is_valid_move(r_idx, new_row, old_row, used_diffs):
                    old_diffs = get_row_diffs(old_row)
                    new_diffs = get_row_diffs(new_row)
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    rows[r_idx] = new_row
                    stuck = 0
                    break

        # Try to reduce the max element of the max row
        target_row = rows[max_r_idx]
        current_max_val = target_row[-1]
        
        # Try to reduce the last element
        lower_bound = target_row[-2] + 1 if k >= 1 else 0
        if lower_bound < current_max_val:
            # Try the smallest possible first
            candidate = lower_bound
            new_row = target_row[:]
            new_row[-1] = candidate
            if is_valid_move(max_r_idx, new_row, target_row, used_diffs):
                old_diffs = get_row_diffs(target_row)
                new_diffs = get_row_diffs(new_row)
                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                rows[max_r_idx] = new_row
                continue
        
        # If last element can't be reduced, try to reduce a previous element
        # to potentially allow a smaller last element later, or just to compact
        if k > 1:
            pos = random.randint(1, k-1)
            old_val = target_row[pos]
            # Try to decrease
            for delta in range(1, 5):
                new_val = old_val - delta
                if new_val <= target_row[pos-1]: break
                if pos < k and new_val >= target_row[pos+1]: continue
                
                new_row = target_row[:]
                new_row[pos] = new_val
                if is_valid_move(max_r_idx, new_row, target_row, used_diffs):
                    old_diffs = get_row_diffs(target_row)
                    new_diffs = get_row_diffs(new_row)
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    rows[max_r_idx] = new_row
                    break

    return best_rows
