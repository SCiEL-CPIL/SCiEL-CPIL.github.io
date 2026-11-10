"""Rewritten program for the suite task; uses Simulated Annealing for DTS."""
import time
import random
import math

def construct(n_rows, k):
    """
    Construct a difference triangle set using Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            if not row or row[0] != 0:
                return False, float('inf')
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry

    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_initial(n_rows, k, rng, perturb=0):
        """
        Greedy initialization with randomization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if time.time() > deadline - 10:
                    return None
                # Find all valid candidates in a window
                candidates = []
                search_limit = x + 20 + (perturb * 10)
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((x, new_diffs))
                    x += 1
                    if time.time() > deadline - 10:
                        return None

                if not candidates:
                    # Expand search if stuck
                    x = search_limit
                    # Force expansion to avoid infinite loop in greedy
                    search_limit += 50
                    continue

                # Choose candidate
                if len(candidates) > 1 and rng.random() < 0.5:
                    # Pick randomly from top few for diversity
                    top = min(3, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def simulated_annealing(rows, n_rows, k, max_time=10.0):
        """
        Apply Simulated Annealing to minimize the scope of the DTS.
        """
        # Deep copy
        current_rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(current_rows)
        current_scope = max(row[-1] for row in current_rows)
        
        best_rows = [row[:] for row in current_rows]
        best_scope = current_scope
        
        start_time_sa = time.time()
        T = 10.0  # Initial temperature
        cooling_rate = 0.995
        min_T = 0.01
        
        rng = random.Random(int(time.time() * 1000) % 10000)
        
        # Pre-calculate total number of elements
        total_elements = n_rows * (k + 1)
        
        while T > min_T and (time.time() - start_time_sa) < max_time:
            if time.time() > deadline - 5:
                break
                
            # Pick a random row and a random position (not the first one)
            r_idx = rng.randint(0, n_rows - 1)
            c_idx = rng.randint(1, k) # 1 to k inclusive
            
            row = current_rows[r_idx]
            old_val = row[c_idx]
            
            # Determine valid range for new_val
            # Must be > row[c_idx-1] and < row[c_idx+1] (if exists)
            min_val = row[c_idx - 1] + 1
            max_val = row[c_idx + 1] - 1 if c_idx < k else current_scope + 50 # Allow expansion
            
            # If range is too large, restrict it to nearby values to keep search local
            # But we also want to allow "jumps" sometimes
            if max_val - min_val > 20:
                # Randomly pick a window
                window_size = rng.randint(5, 20)
                center = rng.randint(min_val, max_val)
                new_min = max(min_val, center - window_size)
                new_max = min(max_val, center + window_size)
            else:
                new_min = min_val
                new_max = max_val
                
            if new_min > new_max:
                T *= cooling_rate
                continue
                
            new_val = rng.randint(new_min, new_max)
            
            if new_val == old_val:
                T *= cooling_rate
                continue
                
            # Check validity
            # Remove old diffs
            old_diffs = set()
            for i in range(c_idx):
                old_diffs.add(old_val - row[i])
            for j in range(c_idx + 1, k + 1):
                old_diffs.add(row[j] - old_val)
                
            temp_used = used_diffs - old_diffs
            
            # Check new diffs
            new_diffs = set()
            valid = True
            for i in range(c_idx):
                d = new_val - row[i]
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                for j in range(c_idx + 1, k + 1):
                    d = row[j] - new_val
                    if d <= 0 or d in temp_used:
                        valid = False
                        break
                    new_diffs.add(d)
                    
            if not valid:
                T *= cooling_rate
                continue
                
            # Calculate new scope
            # Only the last element of a row affects scope if it's the max
            # But changing an internal element doesn't change the last element directly,
            # unless we are changing the last element.
            # Wait, scope is max of all entries.
            # If we change row[r_idx][c_idx], the scope might change if:
            # 1. c_idx == k (last element)
            # 2. The old value was the unique maximum in the whole set
            # 3. The new value is larger than the previous max
            
            # To be safe and simple, recalculate scope only if necessary or just track max.
            # For k=8, n=4, recalculation is cheap.
            # But we can optimize:
            # If c_idx < k, the last element of this row is unchanged.
            # The global max might change if old_val was the global max.
            
            new_scope = current_scope
            if c_idx == k:
                # Last element changed
                # If new_val < old_val, scope might decrease
                # If new_val > old_val, scope increases
                if new_val > old_val:
                    new_scope = max(current_scope, new_val)
                else:
                    # Need to find new max
                    # If old_val was the unique max, we need to find the next max
                    # This is expensive. Let's just calculate max of all rows.
                    # Optimization: keep a list of last elements? No, any element can be max.
                    # For now, calculate max of all elements in the modified row and compare with others?
                    # Actually, scope is max(all entries).
                    # Let's just compute max of current_rows with the modification.
                    # It's O(n*k) which is 4*9=36 ops. Very fast.
                    max_in_row = max(current_rows[r_idx][:c_idx] + [new_val] + current_rows[r_idx][c_idx+1:])
                    max_others = max(max(row) for i, row in enumerate(current_rows) if i != r_idx)
                    new_scope = max(max_in_row, max_others)
            else:
                # Internal element changed
                # If new_val > current_scope, new_scope = new_val
                if new_val > current_scope:
                    new_scope = new_val
                else:
                    # If old_val was the unique global max, scope might decrease.
                    # Check if old_val == current_scope and it's the only one?
                    # Simple check:
                    if old_val == current_scope:
                         # Count how many times current_scope appears
                         count_max = 0
                         for i, row in enumerate(current_rows):
                             if i == r_idx:
                                # Check modified row
                                if new_val == current_scope: count_max += 1
                                # Other elements in this row?
                                for j in range(k+1):
                                    if j != c_idx and row[j] == current_scope:
                                        count_max += 1
                             else:
                                if current_scope in row:
                                    count_max += row.count(current_scope)
                                    # Actually just check if any other row has it
                                    # For simplicity, just recalc if old_val == current_scope
                                    pass
                         # If count_max > 1, scope stays. If 1, scope decreases.
                         # To be safe and fast, just recalc if old_val == current_scope
                         if count_max <= 1:
                             max_in_row = max(current_rows[r_idx][:c_idx] + [new_val] + current_rows[r_idx][c_idx+1:])
                             max_others = max(max(row) for i, row in enumerate(current_rows) if i != r_idx)
                             new_scope = max(max_in_row, max_others)
                    # Else scope stays same
                    else:
                         new_scope = current_scope

            # Acceptance
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept
                current_rows[r_idx][c_idx] = new_val
                used_diffs = temp_used | new_diffs
                current_scope = new_scope
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [row[:] for row in current_rows]
                    
            T *= cooling_rate
            
        return best_rows, best_scope

    # Main search loop
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # 1. Try deterministic greedy
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = det_rows
            best_score = max_entry
            # Run SA on this
            improved, improved_score = simulated_annealing(det_rows, n_rows, k, max_time=15.0)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    # 2. Try randomized restarts with SA
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break

        perturb = rng.randint(0, 10)
        rows = greedy_initial(n_rows, k, rng, perturb=perturb)
        if not rows:
            continue

        valid, max_entry = validate(rows)
        if not valid:
            continue

        if max_entry < best_score:
            best_score = max_entry
            best_result = rows
            # Run SA on this
            improved, improved_score = simulated_annealing(rows, n_rows, k, max_time=10.0)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved
        else:
            # Even if greedy is worse, SA might fix it?
            # Only if we have time.
            if time.time() < deadline - 30:
                improved, improved_score = simulated_annealing(rows, n_rows, k, max_time=5.0)
                if improved_score < best_score:
                    best_score = improved_score
                    best_result = improved

    if best_result is None:
        # Fallback: simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
