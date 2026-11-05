import time
import random
import math
import sys


def _row_diffs_list(row):
    """Return list of all positive differences within a row."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks


def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """
    Greedy construction: build rows one by one, picking the smallest valid value
    for each position with some randomization.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            if max_scope_hint:
                high = max_scope_hint
            else:
                high = 200
            
            candidates = list(range(row[-1] + 1, high + 1))
            
            if len(candidates) > 30:
                sample_size = min(25, len(candidates))
                weights = [1.0 / (1.0 + 0.5 * i) for i in range(len(candidates))]
                selected = []
                pool = list(range(len(candidates)))
                for _ in range(sample_size):
                    total_w = sum(weights[i] for i in pool)
                    r = rng.random() * total_w
                    cum = 0
                    for idx, pi in enumerate(pool):
                        cum += weights[pi]
                        if r <= cum:
                            selected.append(candidates[pi])
                            pool.remove(pi)
                            break
                candidates = sorted(selected)
            
            for val in candidates:
                new_diffs = {val - prev for prev in row}
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                existing_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        existing_diffs.add(row[j] - row[i])
                if new_diffs & existing_diffs:
                    continue
                
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            
            if not found:
                row.append(row[-1] + 10 + rng.randint(0, 20))
                new_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_diffs.add(row[j] - row[i])
                for d in new_diffs:
                    used_mask |= (1 << d)
        
        rows.append(row)
    
    return rows


def _backtrack_row(row_prefix, used_mask, num_elements, max_val, rng, time_limit=None):
    """
    Backtracking to complete a row given a prefix.
    Uses forward checking and randomized candidate ordering.
    Returns the completed row or None if stuck.
    """
    if len(row_prefix) == num_elements:
        return row_prefix
    
    row = row_prefix[:]
    last_val = row[-1]
    
    # Generate candidates
    candidates = list(range(last_val + 1, max_val + 1))
    
    # Shuffle for randomization
    rng.shuffle(candidates)
    
    # Sort by scarcity heuristic: prefer values that leave more flexibility
    # (i.e., values that create fewer conflicts with remaining space)
    def scarcity_score(val):
        # Count how many future positions could still be filled
        # Prefer smaller gaps
        gap = val - last_val
        return gap
    
    # Try candidates in order of scarcity (smallest gap first) with some randomization
    candidates.sort(key=lambda v: scarcity_score(v) + rng.random() * 2)
    
    for val in candidates:
        if time_limit and time.time() > time_limit:
            return None
        
        # Check within-row differences
        new_diffs = set()
        valid = True
        for prev in row:
            d = val - prev
            if d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if not valid:
            continue
        
        # Check against used_mask
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)
        
        if new_mask & used_mask:
            continue
        
        # Forward checking: ensure remaining positions can be filled
        remaining_positions = num_elements - len(row) - 1
        if remaining_positions > 0:
            # Check if there's enough room
            if val + remaining_positions >= max_val + 1:
                continue
        
        # Recurse
        result = _backtrack_row(row + [val], used_mask | new_mask, num_elements, max_val, rng, time_limit)
        if result is not None:
            return result
    
    return None


def _construct_rows_backtracking(n_rows, num_elements, rng, target_scope, time_limit=None):
    """
    Try to construct all rows with maximum value <= target_scope using backtracking.
    Uses a scarcity heuristic to preserve flexibility for later rows.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        if time_limit and time.time() > time_limit:
            return None
        
        row = [0]
        result = _backtrack_row(row, used_mask, num_elements, target_scope, rng, time_limit)
        
        if result is None:
            return None
        
        rows.append(result)
        row_mask = _row_mask(result)
        used_mask |= row_mask
    
    return rows


def _local_search_reduce(rows, num_elements, rng, time_limit):
    """
    Local search: try to reduce scope by moving elements down.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 100000
    stagnation = 0
    max_stagnation = 5000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            continue
        
        new_val = rng.randint(low, old_val - 1)
        
        # Check validity
        row[pos] = new_val
        diffs_in_row = set()
        valid = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs_in_row:
                    valid = False
                    break
                diffs_in_row.add(d)
            if not valid:
                break
        
        if not valid:
            row[pos] = old_val
            stagnation += 1
            continue
        
        new_row_mask = 0
        for d in diffs_in_row:
            new_row_mask |= (1 << d)
        
        other_mask = global_mask & ~row_masks[row_idx]
        if new_row_mask & other_mask:
            row[pos] = old_val
            stagnation += 1
            continue
        
        # Accept
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = (global_mask & ~old_row_mask) | new_row_mask
        
        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            stagnation = 0
        else:
            stagnation += 1
    
    return best_rows, best_scope


def _row_swap_search(rows, num_elements, rng, time_limit):
    """
    Row-swap local search: exchange entire rows to escape element-level local minima.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 50000
    stagnation = 0
    max_stagnation = 3000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        # Pick two rows to swap
        i, j = rng.sample(range(n_rows), 2)
        
        # Try swapping
        current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
        
        if _is_valid(current_rows):
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                # Revert
                current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
                stagnation += 1
        else:
            # Revert
            current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
            stagnation += 1
    
    return best_rows, best_scope


def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=50.0):
    """
    Simulated annealing to minimize scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    temp = initial_temp
    cooling = 0.9995
    min_temp = 0.5
    
    iterations = 0
    max_iter = 200000
    
    while time.time() < time_limit and temp > min_temp and iterations < max_iter:
        iterations += 1
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 10
        
        if low >= high:
            continue
        
        if old_val > low + 2 and rng.random() < 0.7:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        # Check validity
        row[pos] = new_val
        diffs_in_row = set()
        valid = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs_in_row:
                    valid = False
                    break
                diffs_in_row.add(d)
            if not valid:
                break
        
        if not valid:
            row[pos] = old_val
            continue
        
        new_row_mask = 0
        for d in diffs_in_row:
            new_row_mask |= (1 << d)
        
        other_mask = global_mask & ~row_masks[row_idx]
        if new_row_mask & other_mask:
            row[pos] = old_val
            continue
        
        # Accept based on SA criterion
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = (global_mask & ~old_row_mask) | new_row_mask
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            current_rows[row_idx][pos] = old_val
            row_masks[row_idx] = old_row_mask
            global_mask = (global_mask & ~new_row_mask) | old_row_mask
        
        temp *= cooling
    
    return best_rows, best_scope


def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Phase 1: Try backtracking construction at progressively lower scopes
    best_rows = None
    best_scope = float('inf')
    
    # First try with a reasonable scope
    for seed in range(5):
        if time.time() > DEADLINE - 80:
            break
        rng_local = random.Random(seed * 13 + 7)
        # Try scopes from a reasonable upper bound downward
        start_scope = min(200, counting_bound + 50)
        found = False
        for scope in range(start_scope, max(start_scope - 30, counting_bound - 5), -1):
            rows = _construct_rows_backtracking(n_rows, num_elements, rng_local, scope, time.time() + 15)
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                found = True
                break
        
        if found and best_scope <= counting_bound + 10:
            break
    
    # Phase 2: Greedy construction if backtracking didn't find good solutions
    if best_rows is None or best_scope > counting_bound + 20:
        for seed in range(10):
            if time.time() > DEADLINE - 60:
                break
            rng_local = random.Random(seed * 7 + 1)
            rows = _greedy_construct(n_rows, num_elements, rng_local)
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
    
    # If no valid solution found, use a simple sequential fallback
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 40:
        improved_rows, improved_scope = _local_search_reduce(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 30
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Row-swap search
    if time.time() < DEADLINE - 30:
        improved_rows, improved_scope = _row_swap_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 20
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 5: Simulated Annealing
    if time.time() < DEADLINE - 25 and best_scope > counting_bound:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15,
            initial_temp=max(10.0, best_scope * 0.15)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
