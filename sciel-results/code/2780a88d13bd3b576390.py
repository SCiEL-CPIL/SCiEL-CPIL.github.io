import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Multiple greedy restarts to get a good initial solution
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        rows = _greedy_construct(n_rows, k, seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope <= lower_bound:
                break
    
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated annealing with smart moves
    if time.time() < deadline - 10:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Targeted row replacement to reduce scope
    if time.time() < deadline - 5:
        further = _targeted_row_reduction(best_rows, n_rows, k, start_time, deadline)
        if further is not None:
            f_scope = _scope(further)
            if f_scope < best_scope:
                best_scope = f_scope
                best_rows = further
    
    # Phase 4: Final polish - try to reduce each row's max
    if time.time() < deadline - 2:
        polished = _polish_rows(best_rows, n_rows, k, start_time, deadline)
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


def _all_diffs(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _greedy_construct(n_rows, k, seed):
    """Greedy construction with randomization."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            x = last_x + 1
            chosen = None
            
            # Try to find a valid x, preferring smaller values
            candidates = []
            limit = 300
            
            while len(candidates) < 15 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append(x)
                x += 1
            
            if candidates:
                # Weight toward smaller values
                if rng.random() < 0.6:
                    chosen = candidates[0]
                elif rng.random() < 0.8:
                    chosen = candidates[len(candidates) // 2]
                else:
                    chosen = rng.choice(candidates)
            else:
                # Fallback: find smallest valid
                x = last_x + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        chosen = x
                        break
                    x += 1
            
            row.append(chosen)
            new_diffs = {chosen - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows


def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """Simulated annealing to improve the solution."""
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diff sets
    row_diff_sets = [_all_diffs(r) for r in rows]
    
    T = 15.0
    cool_rate = 0.9998
    max_steps = 80000
    
    for step in range(max_steps):
        if time.time() > deadline - 8:
            break
        
        # Pick a random row to modify
        idx = rng.randint(0, n_rows - 1)
        old_row = list(rows[idx])
        
        # Generate a new row candidate
        new_row = _generate_candidate_row(old_row, rows, idx, k, rng, current_scope)
        
        if new_row is None:
            continue
        
        new_diffs = _all_diffs(new_row)
        
        # Check conflicts with other rows
        conflict = False
        for j in range(n_rows):
            if j == idx:
                continue
            if new_diffs & row_diff_sets[j]:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Compute new scope
        new_scope = current_scope
        if new_row[-1] > current_scope:
            new_scope = new_row[-1]
        elif rows[idx][-1] == current_scope and new_row[-1] < current_scope:
            # Need to recompute
            new_scope = max(r[-1] for j, r in enumerate(rows) if j != idx)
            if new_row[-1] > new_scope:
                new_scope = new_row[-1]
        
        delta = new_scope - current_scope
        
        if delta <= 0 or (rng.random() < math.exp(-delta / T)):
            rows[idx] = new_row
            row_diff_sets[idx] = new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        
        T *= cool_rate
    
    return best_rows


def _generate_candidate_row(old_row, all_rows, exclude_idx, k, rng, current_scope):
    """Generate a candidate row that may have lower max value."""
    n = len(old_row)
    max_allowed = current_scope
    
    # Strategy 1: Try to reduce the max value
    if old_row[-1] >= 2:
        target_max = old_row[-1] - rng.randint(1, 3)
        if target_max >= k:  # Must be at least k (0,1,...,k)
            candidate = _construct_row_with_max_limit(
                old_row, target_max, all_rows, exclude_idx, k, rng
            )
            if candidate is not None:
                return candidate
    
    # Strategy 2: Perturb random positions
    for _ in range(3):
        candidate = list(old_row)
        num_mods = rng.randint(1, 3)
        positions = rng.sample(range(1, n), min(num_mods, n - 1))
        
        valid = True
        for pos in positions:
            old_val = candidate[pos]
            delta_val = rng.randint(-4, 4)
            new_val = old_val + delta_val
            
            if new_val <= 0:
                valid = False
                break
            if pos > 0 and new_val <= candidate[pos - 1]:
                valid = False
                break
            if pos < n - 1 and new_val >= candidate[pos + 1]:
                valid = False
                break
            
            candidate[pos] = new_val
        
        if not valid:
            continue
        
        # Check internal distinctness
        if not _is_valid_row(candidate):
            continue
        
        return candidate
    
    return None


def _construct_row_with_max_limit(old_row, max_val, all_rows, exclude_idx, k, rng):
    """Try to construct a row with max <= max_val."""
    row = [0]
    used_in_row = set()
    
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        
        x = last_x + 1
        while x <= max_val:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_in_row):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            return None
        
        # Choose with bias toward smaller values
        if rng.random() < 0.7:
            chosen = valid_xs[0]
        else:
            chosen = rng.choice(valid_xs[:min(8, len(valid_xs))])
        
        row.append(chosen)
        for a in row[:-1]:
            used_in_row.add(chosen - a)
    
    return row


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


def _targeted_row_reduction(initial_rows, n_rows, k, start_time, deadline):
    """Try to reduce the scope by replacing the row(s) with the largest max."""
    rng = random.Random(777)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        # Find the row with the largest max value
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Collect used diffs from other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == max_idx:
                continue
            used_diffs.update(_all_diffs(rows[j]))
        
        # Try to construct a row with smaller max
        for target_max in range(current_max - 1, k - 1, -1):
            if target_max < k:
                break
            
            for attempt in range(5):
                new_row = _construct_row_with_max_limit(
                    rows[max_idx], target_max, rows, max_idx, k, rng
                )
                if new_row is None:
                    continue
                
                new_diffs = _all_diffs(new_row)
                if new_diffs & used_diffs:
                    continue
                
                # Valid replacement
                rows[max_idx] = new_row
                new_scope = _scope(rows)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                
                break  # Found a valid row, move on
        
        # Also try perturbing other rows
        if rng.random() < 0.5:
            idx = rng.randint(0, n_rows - 1)
            new_row = _generate_candidate_row(
                rows[idx], rows, idx, k, rng, best_scope
            )
            if new_row is not None:
                new_diffs = _all_diffs(new_row)
                conflict = False
                for j in range(n_rows):
                    if j == idx:
                        continue
                    if new_diffs & _all_diffs(rows[j]):
                        conflict = True
                        break
                
                if not conflict:
                    rows[idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
    
    return best_rows


def _polish_rows(initial_rows, n_rows, k, start_time, deadline):
    """Final polish: try to reduce each row's max value slightly."""
    rng = random.Random(31415)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Sort rows by max value, try to reduce from largest
    order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
    
    for idx in order:
        if time.time() > deadline - 2:
            break
        
        current_max = rows[idx][-1]
        
        # Try to reduce by 1, 2, 3
        for reduction in [1, 2, 3]:
            target_max = current_max - reduction
            if target_max < k:
                break
            
            used_diffs = set()
            for j in range(n_rows):
                if j == idx:
                    continue
                used_diffs.update(_all_diffs(rows[j]))
            
            new_row = _construct_row_with_max_limit(
                rows[idx], target_max, rows, idx, k, rng
            )
            if new_row is not None:
                new_diffs = _all_diffs(new_row)
                if not (new_diffs & used_diffs):
                    rows[idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    break  # Found a reduction, stop trying larger reductions
    
    return best_rows
