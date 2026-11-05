# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a hybrid approach:
    1. Fast randomized greedy construction to get a good initial upper bound.
    2. Element-level simulated annealing with smooth surrogate cost.
    3. Time-bounded search with safety margins.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
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
    
    # 1. Multi-start Greedy Construction
    def greedy_construct(seed, max_entry=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for _ in range(k):
                last = row[-1]
                remaining_slots = k + 1 - len(row)
                
                if max_entry is not None:
                    max_x = max_entry - (remaining_slots - 1)
                else:
                    max_x = 10000
                
                min_x = last + 1
                
                if min_x > max_x:
                    return None
                
                width = 3
                valid_candidates = []
                
                for x in range(min_x, max_x + 1):
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used or d in local_used:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None
                
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
    
    for seed in range(30):
        if time_up():
            break
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
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            for _ in range(k):
                last = row[-1]
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & local_used):
                        break
                    x += 1
                row.append(x)
                local_used.update(new_diffs)
            rows.append(row)
            used.update(local_used)
        best_rows = rows
        current_rows = rows
        current_used = used
        best_max = get_max(rows)
    
    # Lower bound
    lb = max(k, n_rows * k * (k + 1) // 2)
    
    if best_max == lb:
        return best_rows
    
    # 2. Element-level Simulated Annealing
    rng = random.Random(42)
    current_max = best_max
    
    # Smooth surrogate cost: soft-max of row maxima
    def soft_max_cost(rows, temperature=5.0):
        """
        Compute a smooth surrogate for the max entry using log-sum-exp.
        This provides gradient-like information to guide element moves.
        """
        if not rows:
            return 0.0
        maxima = [max(r) for r in rows]
        max_m = max(maxima)
        if temperature <= 0:
            return float(max_m)
        # log-sum-exp trick for numerical stability
        shifted = [m - max_m for m in maxima]
        log_sum = math.log(sum(math.exp(m / temperature) for m in shifted))
        return max_m + temperature * log_sum
    
    # Try an element-level move: change one element in one row
    def try_element_move(current_rows, current_used, row_idx, elem_idx, max_val_limit):
        """
        Try to change element at position elem_idx in row row_idx to a nearby value.
        Returns (new_row, new_diffs) if successful, else (None, None).
        """
        row = current_rows[row_idx]
        old_val = row[elem_idx]
        
        # Generate candidate values near old_val
        # Must maintain strict increase: row[elem_idx-1] < new_val < row[elem_idx+1]
        min_allowed = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
        max_allowed = row[elem_idx + 1] - 1 if elem_idx < len(row) - 1 else max_val_limit
        
        if min_allowed > max_allowed:
            return None, None
        
        # Try values in a window around old_val
        window = 5
        candidates = []
        for delta in range(0, window + 1):
            for sign in ([1, -1] if delta > 0 else [1]):
                new_val = old_val + sign * delta
                if min_allowed <= new_val <= max_allowed:
                    candidates.append(new_val)
        
        # Shuffle candidates for randomization
        rng.shuffle(candidates)
        
        old_diffs = get_diffs(row)
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            
            # Check if new_val creates valid differences
            new_row = row[:]
            new_row[elem_idx] = new_val
            
            # Verify strict increase
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i - 1]:
                    break
            else:
                new_diffs = get_diffs(new_row)
                
                # Check no conflicts with other rows' diffs
                other_diffs = set()
                for j, r in enumerate(current_rows):
                    if j != row_idx:
                        other_diffs.update(get_diffs(r))
                
                if not (new_diffs & other_diffs):
                    return new_row, new_diffs
        
        return None, None
    
    # Main SA loop
    no_improve_count = 0
    current_temp = 10.0
    cooling_rate = 0.995
    
    while not time_up():
        # Pick a random row and element to try moving
        row_idx = rng.randint(0, n_rows - 1)
        elem_idx = rng.randint(1, k)  # Don't move the first element (must be 0)
        
        old_row = current_rows[row_idx]
        old_diffs = get_diffs(old_row)
        
        # Remove old diffs from global used set
        current_used.difference_update(old_diffs)
        
        new_row, new_diffs = try_element_move(
            current_rows, current_used, row_idx, elem_idx, current_max
        )
        
        if new_row is None:
            current_used.update(old_diffs)
            no_improve_count += 1
            continue
        
        # Check if this move improves or maintains the solution
        current_rows[row_idx] = new_row
        new_global_max = get_max(current_rows)
        
        if new_global_max < current_max:
            # Accept improvement
            current_used.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in current_rows]
            no_improve_count = 0
            current_temp = max(1.0, current_temp * 0.9)  # Cool down on improvement
        elif new_global_max == current_max:
            # Accept with probability based on temperature (Metropolis criterion)
            if rng.random() < 0.3:
                current_used.update(new_diffs)
                no_improve_count += 1
            else:
                # Revert
                current_rows[row_idx] = old_row
                current_used.update(old_diffs)
                no_improve_count += 1
        else:
            # Worse solution - accept with very low probability to escape local minima
            if rng.random() < 0.01:
                current_used.update(new_diffs)
                no_improve_count += 1
            else:
                current_rows[row_idx] = old_row
                current_used.update(old_diffs)
                no_improve_count += 1
        
        # Cool down
        current_temp = max(0.5, current_temp * cooling_rate)
        
        # Periodic restart: if stuck, try a full row rebuild
        if no_improve_count > 200:
            no_improve_count = 0
            # Rebuild the worst row (largest max)
            worst_row_idx = max(range(n_rows), key=lambda i: max(current_rows[i]))
            old_row = current_rows[worst_row_idx]
            old_diffs = get_diffs(old_row)
            current_used.difference_update(old_diffs)
            
            # Rebuild with randomization
            row = [0]
            local_used = set()
            success = True
            for _ in range(k):
                last = row[-1]
                valid_candidates = []
                for x in range(last + 1, current_max + 1):
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in current_used or d in local_used:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= 10:
                            break
                
                if not valid_candidates:
                    success = False
                    break
                
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
                row.append(x)
                local_used.update(new_diffs)
            
            if success:
                new_diffs = get_diffs(row)
                if not (new_diffs & current_used):
                    current_rows[worst_row_idx] = row
                    current_used.update(new_diffs)
                    new_global_max = get_max(current_rows)
                    if new_global_max < current_max:
                        current_max = new_global_max
                        best_rows = [r[:] for r in current_rows]
                else:
                    current_used.update(old_diffs)
            else:
                current_used.update(old_diffs)
    
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
                if row[i] <= row[i - 1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        res = greedy_construct(0)
        if res:
            best_rows, _ = res
    
    return best_rows

# EVOLVE-BLOCK-END
