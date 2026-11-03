import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 110  # Leave 10s margin for 120s limit
    
    # Helper: Greedy construction of a single row
    def build_row(used, limit, randomize=False):
        row = [0]
        # We need to find k more numbers
        for _ in range(k):
            last = row[-1]
            # Try candidates from last+1 up to limit
            # Optimization: We can jump ahead if we know small diffs are used?
            # For now, linear scan is simple.
            valid_candidates = []
            for x in range(last + 1, limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_candidates.append(x)
                    if not randomize:
                        # Greedy: take the first one
                        row.append(x)
                        used.update(new_diffs)
                        break
                    else:
                        # Collect a few valid candidates to choose from
                        # To avoid O(limit) collection, stop after a few
                        if len(valid_candidates) >= 5:
                            break
            if len(row) == k + 1:
                return row
            if randomize and valid_candidates:
                x = random.choice(valid_candidates)
                new_diffs = {x - a for a in row}
                row.append(x)
                used.update(new_diffs)
            else:
                return None
        return row

    # 1. Initial Solution: Deterministic Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        # Find smallest possible max
        # We don't know the limit, so just greedy infinite
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    
    current_max = max(max(r) for r in rows)
    best_max = current_max
    best_rows = [r[:] for r in rows]
    
    # 2. Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Select a row to modify
        # Bias towards the row with the maximum value
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        
        if random.random() < 0.7:
            # Try to shrink the max row
            target_idx = max_idx
            target_limit = rows[target_idx][-1] - 1
        else:
            # Try to shuffle a random row
            target_idx = random.randint(0, n_rows - 1)
            target_limit = rows[target_idx][-1]
            
        # Save state
        old_row = rows[target_idx]
        old_diffs = {old_row[j] - old_row[i] for i in range(k+1) for j in range(i+1, k+1)}
        
        # Remove old diffs
        used.difference_update(old_diffs)
        
        # Try to build new row
        new_row = build_row(used, target_limit, randomize=True)
        
        if new_row is not None:
            new_diffs = {new_row[j] - new_row[i] for i in range(k+1) for j in range(i+1, k+1)}
            # Check if it's actually valid against the rest (it should be by construction, 
            # but build_row updates 'used' locally? No, build_row takes 'used' as read-only 
            # in my implementation above? 
            # Wait, in build_row above, I did `used.update(new_diffs)`. 
            # This modifies the global 'used' set! 
            # I need to be careful. If I fail, I must restore.
            # If I succeed, I keep it.
            
            # Calculate new global max
            # We need to know the max of the *other* rows
            # To avoid O(n_rows) max calculation every time, we can maintain a heap or just compute it.
            # n_rows is likely small (e.g., 7). So O(n_rows) is fine.
            
            # Temporarily set the new row
            rows[target_idx] = new_row
            new_global_max = max(max(r) for r in rows)
            
            if new_global_max < current_max:
                # Always accept improvement
                current_max = new_global_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                # SA acceptance
                delta = new_global_max - current_max
                if random.random() < pow(2.71828, -delta / T):
                    current_max = new_global_max
                else:
                    # Reject: Restore old row and diffs
                    rows[target_idx] = old_row
                    used.difference_update(new_diffs)
                    used.update(old_diffs)
                    # Note: 'used' was modified by build_row. 
                    # If rejected, we must undo the updates made by build_row.
                    # build_row added new_diffs to 'used'.
                    # We removed old_diffs before.
                    # So if rejected:
                    # 1. Remove new_diffs from used.
                    # 2. Add old_diffs to used.
                    # 3. Restore row.
            T *= cooling
        else:
            # Build failed, restore old row
            rows[target_idx] = old_row
            used.update(old_diffs)
            T *= cooling

    return best_rows
