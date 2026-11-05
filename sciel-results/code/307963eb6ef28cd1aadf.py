import time
import random
from collections import Counter

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 elements.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        """Count the number of duplicate differences across all rows."""
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        counts = Counter(diffs)
        conflicts = 0
        for count in counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def objective(rows):
        """Objective: minimize conflicts first, then scope."""
        conflicts = count_conflicts(rows)
        scope = get_scope(rows)
        return conflicts * 1000000 + scope

    def greedy_construct(scope_limit=None, randomize=False, seed=None):
        """
        Greedy construction with optional randomization.
        Returns None if it fails to construct within scope_limit.
        """
        if seed is not None:
            random.seed(seed)
        
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 10000
                
                # Generate candidate values
                if randomize:
                    # Try a window of random values, then sequential
                    window_size = min(50, max_val - last_val)
                    candidates = list(range(last_val + 1, last_val + 1 + window_size))
                    random.shuffle(candidates)
                    # Append the rest sequentially
                    if last_val + 1 + window_size <= max_val:
                        candidates.extend(range(last_val + 1 + window_size, max_val + 1))
                else:
                    candidates = range(last_val + 1, max_val + 1)

                for x in candidates:
                    if x <= last_val:
                        continue
                    if scope_limit is not None and x > scope_limit:
                        break
                    
                    valid = True
                    new_diffs = []
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        for d in new_diffs:
                            used_diffs.add(d)
                        found = True
                        break
                
                if not found:
                    return None
            rows.append(row)
        return rows

    # Step 1: Generate initial solutions using greedy with various scope limits
    best_rows = None
    best_obj = float('inf')

    # Try deterministic greedy with tight scope limits
    for scope_limit in [100, 110, 112, 115, 120, 130, 140, 160, 200, 300, 500, 1000]:
        rows = greedy_construct(scope_limit)
        if rows is not None:
            obj = objective(rows)
            if obj < best_obj:
                best_obj = obj
                best_rows = rows
            # If we found a valid solution with low scope, we can stop early?
            if count_conflicts(rows) == 0 and get_scope(rows) <= 115:
                break

    # If no good solution, try randomized greedy
    if best_rows is None or count_conflicts(best_rows) > 0:
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            seed = random.randint(0, 10000)
            rows = greedy_construct(scope_limit=200, randomize=True, seed=seed)
            if rows is not None:
                obj = objective(rows)
                if obj < best_obj:
                    best_obj = obj
                    best_rows = rows

    # If still no valid solution, use a more generous greedy
    if best_rows is None:
        rows = greedy_construct(scope_limit=None)
        if rows is not None:
            best_rows = rows
            best_obj = objective(rows)
        else:
            # Fallback: construct a trivial valid solution (will be large)
            best_rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    x += 1
                best_rows.append(row)
            best_obj = objective(best_rows)

    # Step 2: Simulated Annealing to refine
    current_rows = [r[:] for r in best_rows]
    current_obj = best_obj
    best_rows_sa = [r[:] for r in best_rows]
    best_obj_sa = best_obj

    temp = 1000.0
    cooling = 0.995
    step_count = 0

    while time.time() < deadline:
        # Generate neighbor
        new_rows = [r[:] for r in current_rows]
        row_idx = random.randint(0, n_rows - 1)
        # Pick an element index > 0 to change
        elem_idx = random.randint(1, k)
        
        low = new_rows[row_idx][elem_idx - 1] + 1
        # Allow expansion up to current best scope + margin
        high = new_rows[row_idx][elem_idx + 1] - 1 if elem_idx < k else best_obj_sa % 1000000 + 10
        
        if low > high:
            continue
            
        new_val = random.randint(low, high)
        
        if new_val == new_rows[row_idx][elem_idx]:
            continue
            
        new_rows[row_idx][elem_idx] = new_val
        
        # Check if row is still strictly increasing
        if not all(new_rows[row_idx][i] < new_rows[row_idx][i+1] for i in range(k)):
            continue
            
        new_obj = objective(new_rows)
        
        delta = new_obj - current_obj
        
        if delta < 0 or (temp > 1.0 and random.random() < pow(2.718, -delta / temp)):
            current_rows = new_rows
            current_obj = new_obj
            
            if current_obj < best_obj_sa:
                best_obj_sa = current_obj
                best_rows_sa = [r[:] for r in current_rows]
                
        temp *= cooling
        step_count += 1

    return best_rows_sa
