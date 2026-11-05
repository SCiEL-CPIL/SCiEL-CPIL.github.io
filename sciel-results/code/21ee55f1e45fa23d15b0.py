import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    num_diffs_per_row = (k + 1) * k // 2
    total_needed = n_rows * num_diffs_per_row

    # Start with a good greedy solution
    best_rows = _greedy_construct(n_rows, k, seed=0)
    best_scope = _scope(best_rows)

    # Try multiple greedy restarts with different seeds
    for seed in range(1, 8):
        if time.time() > deadline - 30:
            break
        candidate = _greedy_construct(n_rows, k, seed=seed)
        scope = _scope(candidate)
        if scope < best_scope:
            best_scope = scope
            best_rows = candidate

    # Backtracking refinement to reduce scope
    if time.time() < deadline - 10:
        refined = _backtrack_refine(best_rows, n_rows, k, start_time, deadline)
        if refined is not None:
            ref_scope = _scope(refined)
            if ref_scope < best_scope:
                best_scope = ref_scope
                best_rows = refined

    # Local search / simulated annealing for further improvement
    if time.time() < deadline - 5:
        improved = _local_search(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    return best_rows


def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)


def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    expected = n_rows * (k + 1) * k // 2
    return len(diffs) == expected


def _greedy_construct(n_rows, k, seed=0):
    """Greedy construction with multiple candidate selection strategies."""
    rng = random.Random(seed)
    used = set()
    rows = []

    for row_idx in range(n_rows):
        row = [0]
        for pos in range(k):
            last_x = row[-1]
            # Try to find the smallest valid x that doesn't conflict
            x = last_x + 1
            max_x = 200 + row_idx * 20  # Allow larger values for later rows
            
            while x <= max_x:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Check if this choice leaves enough room for future elements
                    row.append(x)
                    new_diffs_used = {x - a for a in row[:-1]}
                    
                    # Quick check: can we still complete the row?
                    if _can_complete(row, k, pos + 1, used | new_diffs_used, max_x):
                        used.update(new_diffs_used)
                        break
                    else:
                        row.pop()
                x += 1
            else:
                # Fallback: just pick the first valid x without lookahead
                x = last_x + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1

        rows.append(row)

    return rows


def _can_complete(row, k, pos, used, max_x):
    """Quick check if we can complete the row from current position."""
    if pos >= k:
        return True
    # Just check if there's at least one valid next element
    last_x = row[-1]
    x = last_x + 1
    while x <= max_x:
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            return True
        x += 1
    return False


def _backtrack_refine(rows, n_rows, k, start_time, deadline):
    """Backtracking search to reduce the scope by rebuilding rows."""
    rng = random.Random(123)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Try to rebuild each row to reduce its maximum
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
            
        # Pick a random row to rebuild
        row_idx = rng.randint(0, n_rows - 1)
        
        # Collect differences used by all other rows
        used_diffs = set()
        for i, row in enumerate(best_rows):
            if i == row_idx:
                continue
            n = len(row)
            for a in range(n):
                for b in range(a + 1, n):
                    used_diffs.add(row[b] - row[a])
        
        # Try to build a new row with smaller maximum
        old_max = best_rows[row_idx][-1]
        new_row = _build_row_with_limit(old_max - 1, k, used_diffs, rng)
        
        if new_row is not None:
            # Validate the full configuration
            test_rows = best_rows[:]
            test_rows[row_idx] = new_row
            if _is_valid(test_rows, n_rows, k):
                new_scope = _scope(test_rows)
                if new_scope < best_scope:
                    best_rows = test_rows
                    best_scope = new_scope
    
    return best_rows if best_scope < _scope(rows) else None


def _build_row_with_limit(max_val, k, used_diffs, rng):
    """Build a row of k+1 elements starting at 0, with max value <= max_val."""
    if max_val < k:
        return None
    
    row = [0]
    used_local = set()
    
    for pos in range(k):
        last_x = row[-1]
        candidates = []
        x = last_x + 1
        while x <= max_val:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs) and not (new_diffs & used_local):
                candidates.append(x)
            x += 1
        
        if not candidates:
            return None
        
        # Prefer smaller values but with some randomness
        if rng.random() < 0.7:
            # Pick from the smallest few
            idx = rng.randint(0, min(3, len(candidates) - 1))
            chosen = candidates[idx]
        else:
            chosen = rng.choice(candidates)
        
        row.append(chosen)
        new_diffs = {chosen - a for a in row[:-1]}
        used_local.update(new_diffs)
    
    return row


def _local_search(rows, n_rows, k, start_time, deadline):
    """Local search with row-level moves."""
    rng = random.Random(456)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # Temperature schedule
    T_start = 30.0
    T_end = 0.5
    total_steps = 15000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    T = T_start
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Choose a random row to modify
        row_idx = rng.randint(0, n_rows - 1)
        old_row = list(current_rows[row_idx])
        
        # Generate a new candidate row
        used_diffs = set()
        for i, row in enumerate(current_rows):
            if i == row_idx:
                continue
            n = len(row)
            for a in range(n):
                for b in range(a + 1, n):
                    used_diffs.add(row[b] - row[a])
        
        new_row = _generate_candidate_row_v2(current_rows, row_idx, k, used_diffs, rng)
        
        if new_row is None:
            continue
        
        # Check validity
        test_rows = current_rows[:]
        test_rows[row_idx] = new_row
        if not _is_valid(test_rows, n_rows, k):
            continue
        
        new_scope = _scope(test_rows)
        delta = new_scope - current_scope
        
        if delta <= 0:
            current_rows = test_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_rows = test_rows
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows if best_scope < _scope(rows) else None


def _generate_candidate_row_v2(rows, exclude_idx, k, used_diffs, rng):
    """Generate a new candidate row with bias toward smaller values."""
    row = [0]
    used_local = set()
    
    for pos in range(k):
        last_x = row[-1]
        candidates = []
        x = last_x + 1
        max_x = 150 + pos * 5
        
        while x <= max_x and len(candidates) < 30:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs) and not (new_diffs & used_local):
                candidates.append(x)
            x += 1
        
        if not candidates:
            return None
        
        # Weighted selection favoring smaller values
        weights = [1.0 / (c + 1) for c in candidates]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        chosen = candidates[-1]
        for c, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                chosen = c
                break
        
        row.append(chosen)
        new_diffs = {chosen - a for a in row[:-1]}
        used_local.update(new_diffs)
    
    return row
