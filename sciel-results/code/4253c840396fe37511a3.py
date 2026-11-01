"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 145.0  # safety margin before 150s kill

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

def _validate_solution(rows, n_rows, k):
    """Check that all differences across all rows are distinct and rows are valid."""
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

def _greedy_round_robin(n_rows, k, rng):
    """Round-robin greedy: extend rows one element at a time, choosing the row with smallest last element."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        best_r = -1
        best_last = float('inf')
        order = list(range(n_rows))
        rng.shuffle(order)
        for r in order:
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            if last < best_last:
                best_last = last
                best_r = r
        if best_r == -1:
            return None
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

def _try_greedy_variants(n_rows, k, rng, num_trials=25):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
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
        rows = _greedy_round_robin(n_rows, k, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _local_search_reduce_max(rows, n_rows, k, rng, time_budget):
    """
    Local search focused on reducing the maximum entry.
    Tries to move elements in rows near the max downward.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find rows with elements near the max
        target_rows = [r for r in range(n_rows) if max(rows[r]) >= best_max - 5]
        if not target_rows:
            target_rows = list(range(n_rows))
        
        r = rng.choice(target_rows)
        row = rows[r]
        
        # Try to reduce the last element first
        improved = False
        for pos in [len(row)-1, len(row)-2, rng.randrange(1, len(row))]:
            if pos <= 0:
                continue
            old_val = row[pos]
            # Try smaller values
            for new_val in range(old_val - 1, row[pos-1], -1):
                row_copy = list(row)
                row_copy[pos] = new_val
                new_rd = _compute_diffs_set(row_copy)
                
                # Check internal validity
                if len(new_rd) != k * (k + 1) // 2:
                    continue
                
                # Check against other rows
                conflict = False
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    if new_rd & row_diffs[other_r]:
                        conflict = True
                        break
                
                if not conflict:
                    row[pos] = new_val
                    row_diffs[r] = new_rd
                    cur_max = max(max(rows[rr]) for rr in range(n_rows))
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(rows[rr]) for rr in range(n_rows)]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Random perturbation
            rr = rng.randrange(n_rows)
            pos = rng.randrange(1, len(rows[rr]))
            delta = rng.randint(-3, 3)
            new_val = rows[rr][pos] + delta
            if new_val > rows[rr][pos-1] and (pos >= len(rows[rr])-1 or new_val < rows[rr][pos+1]) and new_val > 0:
                row_copy = list(rows[rr])
                row_copy[pos] = new_val
                new_rd = _compute_diffs_set(row_copy)
                if len(new_rd) == k * (k + 1) // 2:
                    conflict = False
                    for other_r in range(n_rows):
                        if other_r == rr:
                            continue
                        if new_rd & row_diffs[other_r]:
                            conflict = True
                            break
                    if not conflict:
                        rows[rr][pos] = new_val
                        row_diffs[rr] = new_rd
                        cur_max = max(max(rows[x]) for x in range(n_rows))
                        if cur_max < best_max:
                            best_max = cur_max
                            best_rows = [list(rows[x]) for x in range(n_rows)]
    
    return best_rows, best_max

def _iterated_local_search(rows, n_rows, k, rng, time_budget):
    """
    ILS: repeatedly perturb and locally search.
    This is the structural change: instead of one long SA, use kicks + local search.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Initial local search
    rows, cur_max = _local_search_reduce_max(rows, n_rows, k, rng, min(5.0, time_budget * 0.1))
    if cur_max < best_max:
        best_max = cur_max
        best_rows = [list(r) for r in rows]
    
    # ILS loop
    while time.time() - start < time_budget and _time_left() > 5.0:
        # Perturb: change 1-3 random positions
        cur_rows = [list(r) for r in best_rows]
        num_kicks = rng.randint(1, 3)
        for _ in range(num_kicks):
            r = rng.randrange(n_rows)
            pos = rng.randrange(1, len(cur_rows[r]))
            delta = rng.randint(-8, 8)
            new_val = cur_rows[r][pos] + delta
            if new_val > cur_rows[r][pos-1] and (pos >= len(cur_rows[r])-1 or new_val < cur_rows[r][pos+1]) and new_val > 0:
                cur_rows[r][pos] = new_val
        
        # Check validity after perturbation
        if not _validate_solution(cur_rows, n_rows, k):
            # If invalid, try to repair by local search which will fix conflicts
            pass
        
        # Local search from perturbed state
        ls_rows, ls_max = _local_search_reduce_max(cur_rows, n_rows, k, rng, min(3.0, _time_left() - 5.0))
        
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            rows = ls_rows
    
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
    
    # Phase 2: Initial local search
    if _time_left() > 10.0:
        budget = min(15.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Phase 3: Iterated Local Search (structural change)
    if _time_left() > 15.0:
        budget = min(60.0, _time_left() - 5.0)
        ils_rows, ils_max = _iterated_local_search(best_rows, n_rows, k, rng, budget)
        if ils_max < best_max:
            best_max = ils_max
            best_rows = ils_rows
    
    # Phase 4: Final polish
    if _time_left() > 5.0:
        budget = min(10.0, _time_left() - 2.0)
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Final verification
    if not _validate_solution(best_rows, n_rows, k):
        # Fallback to a valid greedy solution
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
    
    return best_rows
