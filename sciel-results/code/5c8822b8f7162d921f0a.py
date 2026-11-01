"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.
Structural change: ILP-style systematic row-reconstruction search via backtracking over
per-row difference-sets, combined with fast greedy initialization and targeted local repair.
"""
import time
import random
import sys
import math

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs_valid(rows, n_rows, k):
    """Check that all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i+1]:
                return False
        d = _compute_diffs_set(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, row_order, rng):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows

def _greedy_interleaved(n_rows, k, rng, strategy="smallest"):
    """Interleaved greedy with multiple strategies."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        
        candidates = [r for r in range(n_rows) if len(rows[r]) < k + 1]
        if not candidates:
            break
        
        if strategy == "smallest":
            best_r = min(candidates, key=lambda r: rows[r][-1])
        elif strategy == "largest":
            best_r = max(candidates, key=lambda r: rows[r][-1])
        elif strategy == "random":
            best_r = rng.choice(candidates)
        elif strategy == "balanced":
            best_r = max(candidates, key=lambda r: len(rows[r]))
        else:
            best_r = min(candidates, key=lambda r: rows[r][-1])
        
        row = rows[best_r]
        x = row[-1] + 1
        while True:
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                placed += 1
                break
            x += 1
    return rows

def _try_greedy_variants(n_rows, k, rng, num_trials=30):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
    strategies = ["smallest", "largest", "random", "balanced"]
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        strategy = strategies[t % len(strategies)]
        rows = _greedy_interleaved(n_rows, k, rng, strategy=strategy)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _backtrack_row(row_prefix, forbidden_diffs, max_val, k, rng, time_limit):
    """
    Backtrack to complete a row given a prefix, forbidden differences, and max value bound.
    Returns a completed row or None.
    """
    if len(row_prefix) == k + 1:
        return list(row_prefix)
    
    if time.time() > time_limit:
        return None
    
    start = row_prefix[-1] + 1
    # Upper bound: we need to place (k+1 - len(row_prefix)) more elements,
    # each at least 1 greater than previous, so max possible is start + (k+1-len(row_prefix)-1)
    # But we also want to keep it small, so we try values in increasing order
    # However for backtracking we might want to try values that allow completion
    
    for x in range(start, max_val + 1):
        # Check if x - a is in forbidden_diffs for all a in row_prefix
        ok = True
        for a in row_prefix:
            d = x - a
            if d in forbidden_diffs:
                ok = False
                break
        if not ok:
            continue
        
        # Check internal consistency: x - a should not equal any existing diff in row_prefix
        # Actually this is covered by forbidden_diffs if we maintain it correctly
        # But we need to check that x-a doesn't conflict with diffs already in row_prefix
        # Wait, forbidden_diffs should include all diffs from other rows AND diffs within current row prefix
        
        # Try extending
        new_prefix = row_prefix + [x]
        # Update forbidden: add x - a for all a in row_prefix
        new_forbidden = set(forbidden_diffs)
        for a in row_prefix:
            new_forbidden.add(x - a)
        
        result = _backtrack_row(new_prefix, new_forbidden, max_val, k, rng, time_limit)
        if result is not None:
            return result
    
    return None

def _reconstruct_row(row_idx, rows, n_rows, k, target_max, rng, time_limit):
    """
    Try to reconstruct row row_idx to have max <= target_max.
    Uses backtracking with the constraint that no diff conflicts with other rows.
    """
    # Compute forbidden diffs from all other rows
    forbidden = set()
    for r in range(n_rows):
        if r == row_idx:
            continue
        forbidden |= _compute_diffs_set(rows[r])
    
    # Start with just [0]
    start_time = time.time()
    # We'll do a limited-depth search with some randomization
    # For efficiency, we use a beam search approach
    
    # Beam search: maintain a set of partial rows
    beam = [([0], set())]  # (row_prefix, diffs_in_prefix)
    
    max_beam_size = 50
    steps_needed = k  # need to add k more elements
    
    for step in range(steps_needed):
        if time.time() > time_limit:
            return None
        if _time_left() < 3.0:
            return None
        
        new_beam = []
        for prefix, prefix_diffs in beam:
            start_val = prefix[-1] + 1
            # Try a range of values
            # We want to keep values small, so try from start_val up to some bound
            # But also need to ensure we can still complete
            remaining_steps = k - step - 1
            # Minimum possible final value if we increment by 1 each time
            min_final = start_val + remaining_steps
            # We want final value <= target_max
            # So start_val + remaining_steps <= target_max
            # => start_val <= target_max - remaining_steps
            max_start = target_max - remaining_steps
            if max_start < start_val:
                continue
            
            # Try values in [start_val, max_start]
            # To keep beam manageable, try a subset
            values_to_try = []
            for x in range(start_val, max_start + 1):
                # Check conflicts
                ok = True
                for a in prefix:
                    d = x - a
                    if d in forbidden or d in prefix_diffs:
                        ok = False
                        break
                if ok:
                    values_to_try.append(x)
            
            # Sort by value (prefer smaller) and take top few
            values_to_try.sort()
            for x in values_to_try[:10]:  # limit to 10 candidates per prefix
                new_prefix = prefix + [x]
                new_diffs = set(prefix_diffs)
                for a in prefix:
                    new_diffs.add(x - a)
                new_beam.append((new_prefix, new_diffs))
        
        if not new_beam:
            return None
        
        # Keep only the best beam_size candidates (smallest last element)
        new_beam.sort(key=lambda item: item[0][-1])
        beam = new_beam[:max_beam_size]
    
    if not beam:
        return None
    
    # Return the best (smallest max)
    beam.sort(key=lambda item: item[0][-1])
    return beam[0][0]

def _systematic_reconstruction(rows, n_rows, k, target_max, rng, time_budget):
    """
    Try to reconstruct each row to have max <= target_max.
    Returns improved rows or None.
    """
    rows = [list(r) for r in rows]
    start = time.time()
    
    # Try to reduce each row one by one
    # Order rows by current max (try to fix the largest first)
    row_order = sorted(range(n_rows), key=lambda r: rows[r][-1], reverse=True)
    
    for r in row_order:
        if time.time() - start > time_budget or _time_left() < 3.0:
            break
        
        # Try to reconstruct row r with max <= target_max
        new_row = _reconstruct_row(r, rows, n_rows, k, target_max, rng, time.time() + 5.0)
        if new_row is not None:
            rows[r] = new_row
    
    # Verify
    if _all_diffs_valid(rows, n_rows, k):
        cur_max = max(max(r) for r in rows)
        if cur_max <= target_max:
            return rows
    
    return None

def _local_search_reduce_max(rows, n_rows, k, rng, time_budget):
    """
    Local search focused on reducing the global maximum.
    Tries to move elements in rows that contain the max value.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        max_val = max(max(rows[r]) for r in range(n_rows))
        if max_val < best_max:
            best_max = max_val
            best_rows = [list(r) for r in rows]
        
        # Find rows with elements near the max
        target_rows = [r for r in range(n_rows) if max(rows[r]) >= max_val - 5]
        if not target_rows:
            break
        
        r = rng.choice(target_rows)
        row = rows[r]
        improved = False
        
        # Try to reduce the last element
        old_val = row[-1]
        for new_val in range(old_val - 1, row[-2], -1):
            new_diffs = set()
            ok = True
            for a in row[:-1]:
                d = new_val - a
                if d in row_diffs[r] or d in set().union(*[row_diffs[rr] for rr in range(n_rows) if rr != r]):
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            
            # Check against other rows
            conflict = False
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                if new_diffs & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                row[-1] = new_val
                row_diffs[r] = _compute_diffs_set(row)
                improved = True
                break
        
        if not improved:
            # Try a random perturbation
            rr = rng.randrange(n_rows)
            pos = rng.randrange(1, len(rows[rr]))
            old = rows[rr][pos]
            delta = rng.randint(-5, 5)
            new = old + delta
            if new <= rows[rr][pos-1] or (pos < k and new >= rows[rr][pos+1]) or new <= 0:
                continue
            
            row_copy = list(rows[rr])
            row_copy[pos] = new
            nd = _compute_diffs_set(row_copy)
            if len(nd) != k * (k + 1) // 2:
                continue
            
            conflict = False
            for other_r in range(n_rows):
                if other_r == rr:
                    continue
                if nd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                rows[rr][pos] = new
                row_diffs[rr] = nd
    
    return best_rows, best_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """SA with conflict counting."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    T = 100.0
    T_min = 0.5
    cooling = 0.998
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        delta = rng.randint(-8, 8)
        new_val = old_val + delta
        
        if new_val <= row[pos-1] or (pos < k and new_val >= row[pos+1]) or new_val <= 0:
            T *= cooling
            continue
        
        row_copy = list(row)
        row_copy[pos] = new_val
        new_rd = _compute_diffs_set(row_copy)
        
        if len(new_rd) != k * (k + 1) // 2:
            T *= cooling
            continue
        
        conflict_count = 0
        for other_r in range(n_rows):
            if other_r == r:
                continue
            conflict_count += len(new_rd & row_diffs[other_r])
        
        if conflict_count == 0:
            row[pos] = new_val
            row_diffs[r] = new_rd
            cur_max = max(max(rows[x]) for x in range(n_rows))
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(rows[x]) for x in range(n_rows)]
        else:
            if rng.random() < 0.005 * (T / 100.0) and conflict_count <= 3:
                row[pos] = new_val
                row_diffs[r] = new_rd
        
        T *= cooling
    
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Greedy with multiple strategies
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=25)
    
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
    
    # Phase 2: Systematic reconstruction with ILP-style backtracking
    if _time_left() > 20.0:
        # Try to reduce max by 1, 2, 3...
        for reduction in range(1, 10):
            if _time_left() < 10.0:
                break
            target = best_max - reduction
            if target < 0:
                break
            new_rows = _systematic_reconstruction(best_rows, n_rows, k, target, rng, min(15.0, _time_left() - 5.0))
            if new_rows is not None:
                new_max = max(max(r) for r in new_rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = new_rows
    
    # Phase 3: Local search
    if _time_left() > 15.0:
        budget = min(25.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Phase 4: Simulated annealing
    if _time_left() > 20.0:
        budget = min(35.0, _time_left() - 5.0)
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Phase 5: Final polish
    if _time_left() > 10.0:
        budget = min(15.0, _time_left() - 3.0)
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Final verification
    if not _all_diffs_valid(best_rows, n_rows, k):
        # Fallback to a valid solution
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
    
    return best_rows
