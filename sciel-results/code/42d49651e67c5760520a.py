"""
Strategy: Structured backtracking search with constraint propagation
for the difference triangle set problem.

Key structural changes from the seed:
1. Backtracking search with MRV (Minimum Remaining Values) heuristic
   for constructing rows, rather than random sampling.
2. Forward checking: when building a row, we check that each new element
   doesn't create duplicate differences within the row or with already-used
   differences globally.
3. Iterative deepening on the scope: try to find solutions with progressively
   smaller maximum values.
4. Local search refinement with bitmask-based difference tracking for speed.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return set of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs


def _row_diffs_list(row):
    """Return list of all positive differences within a row (i<j)."""
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


def _backtrack_row(pos, current_row, used_in_row_mask, global_mask, max_scope, 
                   num_elements, rng, nodes_visited):
    """
    Backtracking search for a single row.
    pos: current position (0 = already set to 0)
    current_row: list being built
    used_in_row_mask: bitmask of differences already in this row
    global_mask: bitmask of differences used in previous rows
    max_scope: maximum allowed value for the last element
    num_elements: total elements in the row (k+1)
    returns: row or None
    """
    if nodes_visited[0] > 500000:
        return None
    
    if pos == num_elements:
        # Check that the last element is <= max_scope
        if current_row[-1] <= max_scope:
            return current_row[:]
        return None
    
    # MRV: for each position, try values that minimize remaining choices
    # We pick the next value greedily with some randomness
    
    if pos == 1:
        # First non-zero element
        low = 1
    else:
        low = current_row[-1] + 1
    
    if pos == num_elements - 1:
        # Last element: must be <= max_scope
        high = max_scope
    else:
        # Leave room for remaining elements
        remaining = num_elements - pos - 1
        high = max_scope - remaining  # at least 1 apart for each remaining
    
    if low > high:
        return None
    
    # Try values in a mixed order: prefer smaller values (to keep scope low)
    # but also try some larger ones for diversity
    candidates = list(range(low, high + 1))
    
    # Shuffle but bias toward smaller values
    # Use a weighted random: smaller values more likely
    if len(candidates) > 20:
        # Sample a subset to limit search
        # Prefer smaller values
        sample_size = min(15, len(candidates))
        # Weighted selection: weight = 1/(1+idx)
        weights = [1.0 / (1.0 + i) for i in range(len(candidates))]
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
        candidates = selected
    
    # Sort candidates: try smaller first (to find compact rows)
    candidates = sorted(candidates)
    
    for val in candidates:
        nodes_visited[0] += 1
        new_row = current_row + [val]
        
        # Compute new differences involving this element
        new_diffs = set()
        for prev in new_row[:-1]:
            new_diffs.add(val - prev)
        
        # Check no duplicates within the row
        if len(new_diffs) != len(set(new_diffs)):
            continue
        
        # Check new diffs don't conflict with already used diffs in this row
        existing_diffs = _row_diffs(new_row[:-1])
        if new_diffs & existing_diffs:
            continue
        
        # Check new diffs don't conflict with global used diffs
        all_new_mask = 0
        for d in new_diffs:
            all_new_mask |= (1 << d)
        
        combined_mask = used_in_row_mask | all_new_mask
        if combined_mask & global_mask:
            continue
        
        result = _backtrack_row(pos + 1, new_row, combined_mask, global_mask,
                                max_scope, num_elements, rng, nodes_visited)
        if result is not None:
            return result
    
    return None


def _construct_rows_backtrack(n_rows, num_elements, max_scope, rng, 
                               time_limit, max_nodes_per_row=200000):
    """
    Construct n_rows using backtracking search for each row.
    """
    start = time.time()
    used_mask = 0
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > time_limit:
            return None
        
        nodes = [0]
        row = _backtrack_row(1, [0], 0, used_mask, max_scope, 
                             num_elements, rng, nodes)
        
        if row is None:
            return None
        
        rows.append(row)
        used_mask |= _row_mask(row)
    
    return rows


def _local_search_v2(rows, num_elements, max_scope, rng, time_limit):
    """
    Local search: try to reduce the maximum value by moving elements.
    Uses bitmask-based difference tracking for speed.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 50000
    stagnation_count = 0
    max_stagnation = 5000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation_count > max_stagnation:
            break
        
        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        row = current_rows[row_idx]
        
        # Pick a position to modify (not the first which is 0)
        pos = rng.randint(1, n_elem - 1)
        old_val = row[pos]
        
        # Determine valid range for new value
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else max_scope
        
        if low >= high:
            continue
        
        # Try to reduce: sample below old_val preferentially
        if old_val > low + 1 and rng.random() < 0.75:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        # Tentatively apply
        row[pos] = new_val
        
        # Check validity
        diffs = _row_diffs_list(row)
        valid = len(diffs) == len(set(diffs))
        
        if valid:
            new_mask = 0
            for d in diffs:
                new_mask |= (1 << d)
            
            other_mask = global_mask & ~row_masks[row_idx]
            valid = (new_mask & other_mask) == 0
        
        if valid:
            # Update masks
            row_masks[row_idx] = new_mask
            global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
            
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                max_scope = best_scope
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            # Revert
            row[pos] = old_val
            row_masks[row_idx] = _row_mask(row)
            global_mask = (global_mask & ~row_masks[row_idx]) | row_masks[row_idx]
            stagnation_count += 1
    
    return best_rows, best_scope


def _sequential_fallback(n_rows, num_elements):
    """
    Sequential greedy construction as a guaranteed fallback.
    """
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
    start_time = time.time()
    DEADLINE = start_time + 135.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Counting bound
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(12345)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try backtracking construction with increasing scope
    # Start from a reasonable lower bound
    initial_scope = max(counting_bound, 80)
    
    # First, get any valid solution using the fallback
    fallback_rows = _sequential_fallback(n_rows, num_elements)
    best_rows = fallback_rows
    best_scope = _scope(fallback_rows)
    
    # Try backtracking from the counting bound upward
    for scope in range(initial_scope, min(best_scope, initial_scope + 60)):
        if time.time() > DEADLINE - 40:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(5):
            rng_local = random.Random(seed * 1000 + scope)
            rows = _construct_rows_backtrack(n_rows, num_elements, scope, 
                                             rng_local, 
                                             time.time() + 10,
                                             max_nodes_per_row=300000)
            if rows is not None:
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                found = True
                break
        
        if found:
            # Found a solution at this scope; try one more level down
            break
    
    # Phase 2: Local search to reduce scope
    remaining_time = DEADLINE - 15
    if remaining_time > 20:
        # Run local search with the best scope as the cap
        improved_rows, improved_scope = _local_search_v2(
            best_rows, num_elements, best_scope, rng,
            time_limit=DEADLINE - 20
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        
        # If we have more time, run another round
        if time.time() < DEADLINE - 25 and best_scope > counting_bound:
            improved_rows2, improved_scope2 = _local_search_v2(
                best_rows, num_elements, best_scope, rng,
                time_limit=DEADLINE - 10
            )
            if improved_scope2 < best_scope:
                best_scope = improved_scope2
                best_rows = improved_rows2
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        return _sequential_fallback(n_rows, num_elements)

# EVOLVE-BLOCK-END
