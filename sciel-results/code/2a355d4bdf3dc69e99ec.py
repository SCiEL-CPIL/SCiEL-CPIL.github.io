import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses Simulated Annealing with row-rebuilding moves to minimize the maximum entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Greedy Initialization
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows, used
    
    rows, used = greedy_construct()
    current_max = max(max(r) for r in rows)
    best_rows = rows[:]
    best_max = current_max
    
    # Helper to get diffs of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to rebuild a row with a given max limit and randomization
    # Returns (new_row, new_diffs) or (None, None)
    def rebuild_row(row_idx, max_limit, rng):
        # Remove current row diffs from global 'used'
        old_row = rows[row_idx]
        old_diffs = get_diffs(old_row)
        for d in old_diffs:
            used.remove(d)
        
        new_row = [0]
        success = True
        
        while len(new_row) < k + 1:
            last = new_row[-1]
            # We need to pick x > last such that x <= max_limit
            # And diffs (x - a) for a in new_row are not in 'used'
            
            # Determine valid candidates
            # x can be at most max_limit
            # But we also need to be able to fit remaining (k+1 - len(new_row)) elements
            # min gap is 1, so x <= max_limit - (remaining - 1)
            remaining = k + 1 - len(new_row)
            max_x = max_limit - (remaining - 1)
            
            if last >= max_x:
                success = False
                break
            
            candidates = []
            # To speed up, we can just iterate x from last+1 to max_x
            # But if max_limit is large, this is slow.
            # However, k is small (7), so row length is 8.
            # The number of steps is small.
            
            # Heuristic: Prefer smaller x? Or random?
            # For SA, we want to explore.
            # Let's pick randomly from valid candidates in [last+1, max_x].
            # To bias towards packing, we might weight smaller x higher.
            
            valid_xs = []
            for x in range(last + 1, max_x + 1):
                valid = True
                new_diffs = []
                for a in new_row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    valid_xs.append((x, new_diffs))
            
            if not valid_xs:
                success = False
                break
            
            # Select a candidate
            # Simple random choice
            # Or weighted: weight = 1 / x
            # For now, random choice from valid candidates
            # To encourage smaller values, we can sort and pick from the first few?
            # No, pure random allows escaping local minima.
            # But we want to find *small* max entries.
            # If we are trying to reduce max, we should bias towards smaller x.
            # Let's use a probability distribution that favors smaller x.
            # P(x) proportional to 1/x
            
            # If we just want to find *any* valid row, random is fine.
            # If we want to minimize max, we should try to keep x small.
            # The max_limit already constrains the end.
            # Let's just pick the smallest valid x? That's greedy.
            # SA needs randomness.
            # Let's pick randomly from the first 10 valid candidates?
            # Or just random from all?
            # Given the tight constraints, the number of valid candidates might be small.
            
            # Let's pick randomly from all valid candidates.
            x, new_diffs = valid_xs[rng.randint(0, len(valid_xs) - 1)]
            
            new_row.append(x)
            for d in new_diffs:
                used.add(d)
        
        if success:
            rows[row_idx] = new_row
            return True
        else:
            # Restore old row
            rows[row_idx] = old_row
            for d in old_diffs:
                used.add(d)
            return False

    # Simulated Annealing Loop
    rng = random.Random(42)
    T = 10.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        # Pick a row to rebuild
        # Bias towards rows with large max entries
        # Find max entry
        max_val = 0
        max_indices = []
        for i, r in enumerate(rows):
            if max(r) > max_val:
                max_val = max(r)
                max_indices = [i]
            elif max(r) == max_val:
                max_indices.append(i)
        
        # 50% chance to pick a max row, 50% chance to pick a random row
        if rng.random() < 0.5 and max_indices:
            idx = rng.choice(max_indices)
        else:
            idx = rng.randint(0, n_rows - 1)
        
        # Determine max_limit for rebuild
        # If we picked a max row, try to reduce it
        # If we picked a random row, try to keep it same or smaller
        row_max = max(rows[idx])
        
        if row_max == max_val:
            # Try to reduce
            # Try max_limit = row_max - 1
            # If fails, try row_max
            max_limit = row_max - 1
        else:
            max_limit = row_max
        
        # Attempt rebuild
        # We need to calculate the new global max if successful
        # But rebuild_row modifies 'rows' and 'used' in place.
        # We need to be able to revert if we reject the move.
        
        # Let's snapshot the state
        old_rows = rows[:]
        old_used = used.copy()
        old_max = current_max
        
        # Try rebuild
        # We might need to try multiple times with different max_limits?
        # Let's just try with the chosen max_limit.
        
        # Optimization: If max_limit < 0, skip
        if max_limit < 0:
            continue
            
        success = rebuild_row(idx, max_limit, rng)
        
        if success:
            new_rows = rows[:]
            new_used = used.copy()
            new_max = max(max(r) for r in new_rows)
            
            delta = new_max - old_max
            
            # Acceptance criterion
            if delta < 0:
                # Improvement
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = new_rows[:]
            else:
                # No improvement or worse
                # Accept with probability exp(-delta / T)
                if delta == 0:
                    # Neutral move, accept with some probability to shuffle
                    if rng.random() < 0.5:
                        current_max = new_max
                        # Don't update best_rows unless it's better, which it isn't
                else:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        current_max = new_max
                    else:
                        # Reject: Restore state
                        rows = old_rows
                        used = old_used
                        current_max = old_max
        else:
            # Rebuild failed, state is already restored by rebuild_row
            pass
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Final validation
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

    if not validate(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows, _ = greedy_construct()
        
    return best_rows
