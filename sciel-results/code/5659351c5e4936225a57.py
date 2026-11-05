import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Improved Greedy with Backtracking to get an initial upper bound
    best_rows = _construct_with_backtracking(n_rows, k, 0, deadline)
    if not best_rows:
        # Fallback to simple greedy if backtracking fails
        best_rows = _greedy_construct(n_rows, k, 0)
    best_scope = _scope(best_rows)
    
    # Phase 2a: Multiple randomized restarts with backtracking
    for seed in range(1, 100):
        if time.time() > deadline - 30:
            break
        rows = _construct_with_backtracking(n_rows, k, seed, deadline)
        if rows:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lower_bound:
                    break
    
    # Phase 2b: Local search with packing-aware moves
    if time.time() < deadline - 5:
        improved = _local_search_packing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 2c: Final polish pass - exhaustively try to shrink each row
    if time.time() < deadline - 1:
        polished = _polish_pass(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            p_scope = _scope(polished)
            if p_scope < best_scope:
                best_scope = p_scope
                best_rows = polished
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _packing_score(row):
    """
    Score a row for packing quality.
    Lower is better.
    Combines the max value and the sum of differences (proxy for density/spread).
    """
    if not row:
        return 0
    max_val = row[-1]
    diffs = _row_diffs(row)
    # Sum of diffs is a measure of "waste" if we consider the ideal dense set
    # Actually, smaller sum of diffs usually correlates with smaller max_val for fixed k
    # Let's use max_val * 100 + sum(diffs) to prioritize scope first, then density
    return max_val * 100 + sum(diffs)

def _greedy_construct(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            last_x = row[-1]
            limit = 500
            
            x = last_x + 1
            while len(valid_xs) < 20 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 20:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            if not valid_xs:
                chosen_x = last_x + 1
                while True:
                    new_diffs = {chosen_x - a for a in row}
                    if not (new_diffs & used):
                        break
                    chosen_x += 1
            else:
                chosen_x = rng.choice(valid_xs)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _construct_with_backtracking(n_rows, k, seed, deadline):
    """
    Construct rows using a backtracking search with a packing heuristic.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        # We will build the row element by element
        # If we get stuck or the score is bad, we might backtrack (not fully implemented here for speed, 
        # but we use a heuristic to pick the "best" next step)
        
        for _ in range(k):
            last_x = row[-1]
            candidates = []
            
            # Look ahead a bit to find valid candidates
            # We want candidates that keep the max small and differences dense
            x = last_x + 1
            # Limit search space to avoid explosion
            # Estimate max possible x: if we need k more steps, and min step is 1, max is last_x + k * max_step
            # But we want small max. Let's search up to last_x + 50 or so initially
            # If we don't find any, expand.
            
            found_any = False
            search_limit = last_x + 100
            while x <= search_limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Heuristic: prefer smaller x
                    candidates.append(x)
                    if len(candidates) > 10:
                        break
                x += 1
                found_any = True
                
            if not candidates:
                # Expand search
                x = search_limit + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) > 10:
                            break
                    x += 1
            
            if not candidates:
                return None # Failure
            
            # Choose the candidate that minimizes the packing score of the partial row
            # Or just pick the smallest to keep scope low?
            # Smallest x is usually best for scope.
            # Let's pick the smallest valid x.
            chosen_x = min(candidates)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            
            # Check time
            if time.time() > deadline - 10:
                return None
        
        rows.append(row)
        
    return rows

def _local_search_packing(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search that optimizes both scope and packing score.
    """
    rng = random.Random(12345)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diff sets
    row_diff_sets = [_row_diffs(r) for r in rows]
    
    max_steps = 50000
    T = 50.0 # Higher temperature to allow packing improvements
    cool_rate = 0.999
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        idx = rng.randint(0, n_rows - 1)
        old_row = list(rows[idx])
        
        # Generate a perturbed row
        # Strategy: Pick a random position and try to change it to a smaller value
        # or shift a segment
        new_row = _perturb_row_packing(old_row, rows, idx, k, rng)
        
        if new_row is None:
            continue
        
        new_diffs = _row_diffs(new_row)
        
        # Check conflicts
        conflict = False
        for j in range(n_rows):
            if j == idx:
                continue
            if new_diffs & row_diff_sets[j]:
                conflict = True
                break
        
        if conflict:
            continue
        
        new_rows = rows[:]
        new_rows[idx] = new_row
        new_scope = _scope(new_rows)
        
        # Calculate score change
        # We want to minimize scope. If scope is same, we prefer better packing (lower sum of diffs)
        old_score = current_scope * 100 + sum(_row_diffs(old_row))
        new_score = new_scope * 100 + sum(new_diffs)
        
        delta = new_score - old_score
        
        if delta <= 0:
            # Accept
            rows = new_rows
            row_diff_sets[idx] = new_diffs
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Metropolis
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                rows = new_rows
                row_diff_sets[idx] = new_diffs
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows

def _perturb_row_packing(old_row, all_rows, exclude_idx, k, rng):
    """
    Generate a perturbed row trying to reduce scope or improve packing.
    """
    row = list(old_row)
    n = len(row)
    
    # Try to reduce the maximum element
    if rng.random() < 0.5:
        # Try to move the last element down
        pos = n - 1
        old_val = row[pos]
        # Try values from old_val - 1 down to row[pos-1] + 1
        for new_val in range(old_val - 1, row[pos-1], -1):
            candidate = list(row)
            candidate[pos] = new_val
            if _is_valid_row(candidate):
                return candidate
    else:
        # Try to move a random internal element down
        pos = rng.randint(1, n - 1)
        old_val = row[pos]
        for new_val in range(old_val - 1, row[pos-1], -1):
            candidate = list(row)
            candidate[pos] = new_val
            # Check if subsequent elements still valid (they are, since we only decreased)
            # But we need to check internal diffs
            if _is_valid_row(candidate):
                return candidate
    return None

def _is_valid_row(row):
    """Check if a row is strictly increasing with distinct internal differences."""
    n = len(row)
    for i in range(n - 1):
        if row[i + 1] <= row[i]:
            return False
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _polish_pass(initial_rows, n_rows, k, start_time, deadline):
    """
    Final polish: try to reduce the max of each row individually.
    """
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for idx in range(n_rows):
        if time.time() > deadline - 1:
            break
        
        old_row = rows[idx]
        old_max = old_row[-1]
        
        # Collect diffs from other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            used_diffs.update(_row_diffs(rows[j]))
        
        # Try to find a row with max < old_max
        # We can try to "shift" the entire row down if possible? No, it must start at 0.
        # So we must reconstruct.
        # Let's try to reduce the last element first.
        # If we reduce row[-1], we might need to adjust previous elements.
        
        # Simple approach: Try to replace the row with a "tighter" one
        # using a greedy method with a tighter limit.
        
        for target_max in range(old_max - 1, k - 1, -1):
            if time.time() > deadline - 1:
                break
            
            # Try to construct a row with max <= target_max
            new_row = _construct_row_with_limit(used_diffs, k, target_max, random.Random(42))
            if new_row is not None:
                rows[idx] = new_row
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                break
        
    return best_rows

def _construct_row_with_limit(used_diffs, k, max_val, rng):
    """
    Construct a row of k+1 elements starting at 0, max <= max_val,
    with all internal diffs avoiding used_diffs.
    """
    row = [0]
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        
        x = last_x + 1
        while x <= max_val:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            return None
        
        # Choose the smallest valid x to keep max low
        chosen = valid_xs[0]
        
        row.append(chosen)
        for a in row[:-1]:
            used_diffs.add(chosen - a)
    
    return row
