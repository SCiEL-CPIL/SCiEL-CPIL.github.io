import time
import random
import math

_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    """Return set of all pairwise differences in a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _build_diff_counts(rows):
    """Build a dict mapping each difference to the count of rows containing it."""
    counts = {}
    for r in rows:
        for d in _row_diffs(r):
            counts[d] = counts.get(d, 0) + 1
    return counts

def _remove_row_diffs(counts, row):
    """Remove one row's differences from the count dict."""
    for d in _row_diffs(row):
        counts[d] -= 1
        if counts[d] == 0:
            del counts[d]

def _add_row_diffs(counts, row):
    """Add one row's differences to the count dict."""
    for d in _row_diffs(row):
        counts[d] = counts.get(d, 0) + 1

def _get_cost(rows, diff_counts):
    """Compute cost: scope + penalty for any duplicate differences."""
    scope = _scope(rows)
    penalty = sum(max(0, c - 1) for c in diff_counts.values())
    return scope + 1000 * penalty

def _fast_greedy(n_rows, k, rng, aggressiveness=0.3, window=8):
    """
    Randomized greedy: for each position, consider a window of candidates
    and pick randomly from valid ones, biased toward smaller values.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            best_candidates = []
            x = last + 1
            for _ in range(window):
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    best_candidates.append((x, new_diffs))
                    if len(best_candidates) >= 4:
                        break
                x += 1
            
            if not best_candidates:
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        best_candidates.append((x, new_diffs))
                        break
                    x += 1
            
            # Weighted random selection biased toward smaller values
            weights = [1.0 / (c[0] - last + 1) ** 1.2 for c in best_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = best_candidates[0]
            for c, w in zip(best_candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _csp_build_row(k, used_diffs, rng, max_val=200):
    """
    Build a single row of k+1 elements starting at 0, avoiding used_diffs.
    Returns the row or None if impossible within max_val.
    """
    row = [0]
    last = 0
    
    for j in range(1, k + 1):
        found = False
        for x in range(last + 1, max_val + 1):
            valid = True
            new_diffs = []
            for i in range(len(row)):
                d = x - row[i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                last = x
                found = True
                break
        if not found:
            return None
    return row

def _sa_optimize(rows, k, deadline, rng):
    """
    Simulated annealing to minimize scope while maintaining validity.
    """
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    
    # Build initial diff counts
    diff_counts = _build_diff_counts(rows)
    scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = scope
    best_cost = _get_cost(rows, diff_counts)
    
    T = 8.0
    cooling = 0.9998
    min_T = 0.05
    
    while time.time() < deadline:
        # Pick a random element to move
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = rows[i][j]
        prev_val = rows[i][j - 1]
        next_val = rows[i][j + 1] if j < k else scope + 200
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        # Propose new value
        if rng.random() < 0.5:
            # Small perturbation
            span = hi - lo
            delta = int(rng.gauss(0, max(1, span * 0.12)))
            new_val = old_val + delta
        else:
            new_val = rng.randint(lo, hi)
        
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        # Compute cost change
        # Remove old diffs involving rows[i][j]
        old_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(rows[i][t] - old_val)
                old_diffs.add(d)
        
        # Check new diffs
        new_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(rows[i][t] - new_val)
                new_diffs.add(d)
        
        # Check collisions: new_diffs that are already in diff_counts (after removing old_diffs)
        collisions = 0
        for d in new_diffs:
            count = diff_counts.get(d, 0)
            # Subtract the contribution from old_diffs if d was in old_diffs
            if d in old_diffs:
                count -= 1
            if count > 0:
                collisions += 1
        
        # New scope
        new_scope = max(scope, new_val)
        if old_val == scope and new_val < old_val:
            max_s = 0
            for r in rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            new_scope = max_s
        
        old_cost = scope + 1000 * 0
        new_cost = new_scope + 1000 * collisions
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            # Accept
            # Update diff_counts
            _remove_row_diffs(diff_counts, rows[i])
            rows[i][j] = new_val
            _add_row_diffs(diff_counts, rows[i])
            scope = new_scope
            
            current_cost = _get_cost(rows, diff_counts)
            if collisions == 0 and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                best_cost = current_cost
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _local_polish(rows, k, deadline, rng):
    """
    Local search to minimize scope while maintaining validity.
    Moves: change a single element in a row to a nearby value.
    """
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    
    diff_counts = _build_diff_counts(rows)
    scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = scope
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline:
        # Pick a random element to move
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = rows[i][j]
        prev_val = rows[i][j - 1]
        next_val = rows[i][j + 1] if j < k else scope + 100
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        # Propose new value
        if rng.random() < 0.6:
            span = hi - lo
            delta = int(rng.gauss(0, max(1, span * 0.15)))
            new_val = old_val + delta
        else:
            new_val = rng.randint(lo, hi)
        
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        # Compute cost change
        old_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(rows[i][t] - old_val)
                old_diffs.add(d)
        
        new_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(rows[i][t] - new_val)
                new_diffs.add(d)
        
        # Check collisions
        collisions = 0
        for d in new_diffs:
            count = diff_counts.get(d, 0)
            if d in old_diffs:
                count -= 1
            if count > 0:
                collisions += 1
        
        # New scope
        new_scope = max(scope, new_val)
        if old_val == scope and new_val < old_val:
            max_s = 0
            for r in rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            new_scope = max_s
        
        old_cost = scope + 1000 * 0
        new_cost = new_scope + 1000 * collisions
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            # Accept
            _remove_row_diffs(diff_counts, rows[i])
            rows[i][j] = new_val
            _add_row_diffs(diff_counts, rows[i])
            scope = new_scope
            
            if collisions == 0 and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts
    best_rows = None
    best_scope = float('inf')
    
    n_restarts = 80
    deadline_phase1 = start_time + 40.0
    
    for restart in range(n_restarts):
        if time.time() > deadline_phase1:
            break
        
        # Vary parameters
        aggr = 0.1 + 0.8 * (restart % 10) / 10.0
        window = 6 + (restart % 5) * 2
        
        rows = _fast_greedy(n_rows, k, rng, aggressiveness=aggr, window=window)
        
        if _valid(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            for j in range(1, k + 1):
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        last = x
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 2: Simulated annealing to improve
    remaining = _DEADLINE - time.time()
    if remaining > 20 and best_rows is not None:
        ls_scope, ls_rows = _sa_optimize(best_rows, k, time.time() + remaining * 0.6, rng)
        
        if _valid(ls_rows, k):
            ls_scope = _scope(ls_rows)
            if ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = ls_rows
    
    # Phase 3: Local polish
    remaining = _DEADLINE - time.time()
    if remaining > 10 and best_rows is not None:
        lp_scope, lp_rows = _local_polish(best_rows, k, time.time() + remaining * 0.8, rng)
        
        if _valid(lp_rows, k):
            lp_scope = _scope(lp_rows)
            if lp_scope < best_scope:
                best_scope = lp_scope
                best_rows = lp_rows
    
    return best_rows
