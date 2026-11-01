"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.

Stronger algorithm combining:
1. Greedy baseline with multiple orderings
2. Backtracking with MRV, forward checking, and binary search on max value
3. Local search with efficient conflict tracking
4. Iterated Local Search to escape local minima
"""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _diffs_of_row(row):
    """Return a set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _validate(rows):
    """Check all differences are distinct across all rows. Return True if valid."""
    seen = set()
    for row in rows:
        d = _diffs_of_row(row)
        if len(d) != len(row) * (len(row) - 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True


def _greedy_build(n_rows, k, rng, order=None):
    """Build rows greedily in the given order."""
    if order is None:
        order = list(range(n_rows))
    used = set()
    rows = [None] * n_rows
    for r in order:
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


def _local_search_reduce(rows, n_rows, k, rng, time_budget):
    """
    Local search: try to reduce the maximum value by moving elements down.
    Returns (best_rows, best_max).
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()

    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]

    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the maximum element
        max_row = -1
        max_val = -1
        for r in range(n_rows):
            if rows[r][-1] > max_val:
                max_val = rows[r][-1]
                max_row = r
        
        if max_row == -1:
            break
        
        row = rows[max_row]
        improved = False
        
        # Try to reduce the last element
        for new_val in range(row[-1] - 1, row[-2], -1):
            row_copy = list(row)
            row_copy[-1] = new_val
            new_rd = _diffs_of_row(row_copy)
            if len(new_rd) != k * (k + 1) // 2:
                continue
            conflict = False
            for other_r in range(n_rows):
                if other_r == max_row:
                    continue
                if new_rd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                row[-1] = new_val
                row_diffs[max_row] = new_rd
                new_max = max(max(rows[x]) for x in range(n_rows))
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
                improved = True
                break
        
        if not improved:
            # Try reducing a random non-last element in a random row
            r = rng.randrange(n_rows)
            if len(rows[r]) <= 2:
                break
            pos = rng.randrange(1, len(rows[r]) - 1)
            row = rows[r]
            old = row[pos]
            for new_val in range(old - 1, row[pos - 1], -1):
                if pos < k and new_val >= row[pos + 1]:
                    continue
                row_copy = list(row)
                row_copy[pos] = new_val
                new_rd = _diffs_of_row(row_copy)
                if len(new_rd) != k * (k + 1) // 2:
                    continue
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
                    new_max = max(max(rows[x]) for x in range(n_rows))
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(rows[x]) for x in range(n_rows)]
                    break

    return best_rows, best_max


def _backtrack_construct(n_rows, k, max_val, rng, time_budget=10.0):
    """
    Backtracking with forward checking and MRV to construct rows.
    Each row is built by choosing values that minimize future conflicts.
    """
    start = time.time()
    used = set()
    rows = [None] * n_rows
    
    order = list(range(n_rows))
    rng.shuffle(order)
    
    def can_complete(row, used_so_far, remaining_positions, current_used):
        """Check if we can still complete the row given current state."""
        if time.time() - start > time_budget:
            return False
        if len(row) >= k + 1:
            return True
        last = row[-1]
        # Check if there are enough available values
        available = 0
        for val in range(last + 1, max_val + 1):
            ok = True
            for a in row:
                d = val - a
                if d in used_so_far or d in current_used:
                    ok = False
                    break
            if ok:
                available += 1
                if available >= remaining_positions:
                    return True
        return available >= remaining_positions
    
    def build_row(used_so_far):
        """Build one row given the current used set, using MRV and forward checking."""
        if time.time() - start > time_budget:
            return None
        
        row = [0]
        current_used = set()
        
        def extend(current_row, current_used_set):
            if time.time() - start > time_budget:
                return None
            
            if len(current_row) == k + 1:
                return list(current_row)
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Collect all valid next values
            candidates = []
            for val in range(last + 1, max_val + 1):
                ok = True
                new_diffs = set()
                for a in current_row:
                    d = val - a
                    if d in current_used_set or d in used_so_far:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    # Score: how many future values this blocks
                    block_count = 0
                    for a2 in current_row + [val]:
                        for d2 in new_diffs:
                            if d2 in used_so_far:
                                block_count += 1
                    candidates.append((val, len(new_diffs)))
            
            if not candidates:
                return None
            
            # Sort by number of differences (fewer = better, leaves more room)
            candidates.sort(key=lambda x: x[1])
            
            # Try candidates in order
            for val, _ in candidates:
                new_diffs = set()
                for a in current_row:
                    new_diffs.add(val - a)
                
                result = extend(current_row + [val], current_used_set | new_diffs)
                if result is not None:
                    return result
            
            return None
        
        return extend(row, current_used)
    
    for r in order:
        if time.time() - start > time_budget:
            return None
        
        row = build_row(used)
        if row is None:
            return None
        rows[r] = row
        used |= _diffs_of_row(row)
    
    return rows


def _sa_optimize(rows, n_rows, k, rng, time_budget):
    """
    Simulated annealing to reduce the maximum entry.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)

    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]

    start = time.time()
    T = max(1.0, best_max * 0.1)
    T_min = 0.5
    cooling = 0.997

    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        r = rng.randrange(n_rows)
        row = rows[r]
        if len(row) <= 1:
            T *= cooling
            continue
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        span = max(1, min(15, int(T)))
        delta = rng.randint(-span, span)
        new_val = old_val + delta
        
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        row_copy = list(row)
        row_copy[pos] = new_val
        new_rd = _diffs_of_row(row_copy)
        
        if len(new_rd) != k * (k + 1) // 2:
            T *= cooling
            continue
        
        conflict = 0
        for other_r in range(n_rows):
            if other_r == r:
                continue
            conflict += len(new_rd & row_diffs[other_r])
        
        if conflict == 0:
            cur_max = 0
            for x in range(n_rows):
                if x == r:
                    for v in row_copy:
                        if v > cur_max:
                            cur_max = v
                else:
                    for v in rows[x]:
                        if v > cur_max:
                            cur_max = v
            
            if cur_max <= best_max or rng.random() < 0.1:
                row[pos] = new_val
                row_diffs[r] = new_rd
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
        else:
            if conflict <= 2 and rng.random() < 0.01 * (T / 100.0):
                row[pos] = new_val
                row_diffs[r] = new_rd

        T *= cooling

    return best_rows, best_max


def _iterated_local_search(rows, n_rows, k, rng, time_budget):
    """
    Iterated Local Search: repeatedly perturb and run local search.
    """
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in best_rows)
    start = time.time()
    
    current_rows = [list(r) for r in rows]
    current_max = best_max
    
    max_iter = 200
    per_iter_time = time_budget / max_iter if max_iter > 0 else 1.0
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Kick: perturb the current solution
        num_kicks = rng.randint(2, 5)
        for _ in range(num_kicks):
            r = rng.randrange(n_rows)
            row = current_rows[r]
            if len(row) <= 1:
                continue
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            span = rng.randint(3, 10)
            new_val = old_val + rng.randint(-span, span)
            
            if new_val <= row[pos - 1] or new_val <= 0:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue
            
            row[pos] = new_val
        
        if not _validate(current_rows):
            current_rows = [list(r) for r in best_rows]
            current_max = best_max
            continue
        
        ls_rows, ls_max = _local_search_reduce(current_rows, n_rows, k, rng, per_iter_time)
        
        if ls_max < best_max:
            best_max = ls_max
            best_rows = [list(r) for r in ls_rows]
            current_rows = [list(r) for r in ls_rows]
            current_max = best_max
        else:
            current_rows = [list(r) for r in ls_rows]
            current_max = ls_max
        
        if it % 10 == 0 and it > 0:
            current_rows = [list(r) for r in best_rows]
            current_max = best_max

    return best_rows, best_max


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Greedy baseline with several orderings
    num_greedy = 20
    for t in range(num_greedy):
        if _time_left() < 5.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, rng, order)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)

    # Phase 2: Backtracking with binary search on max_val
    if _time_left() > 30.0:
        total_diffs = n_rows * k * (k + 1) // 2
        lo = max(total_diffs, k)
        hi = best_max
        
        bt_start = time.time()
        while lo <= hi and _time_left() > 10.0 and time.time() - bt_start < 60.0:
            mid = (lo + hi) // 2
            if _time_left() < 5.0:
                break
            
            bt_rows = _backtrack_construct(n_rows, k, mid, rng, time_budget=8.0)
            if bt_rows is not None:
                if _validate(bt_rows):
                    cur_max = max(max(r) for r in bt_rows)
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(r) for r in bt_rows]
                    hi = mid - 1
                else:
                    hi = mid - 1
            else:
                lo = mid + 1
    
    # Phase 3: Iterated Local Search
    if _time_left() > 20.0:
        budget = min(60.0, _time_left() - 10.0)
        ils_rows, ils_max = _iterated_local_search(best_rows, n_rows, k, rng, budget)
        if ils_max < best_max:
            best_max = ils_max
            best_rows = ils_rows

    # Phase 4: Simulated annealing polish
    if _time_left() > 15.0:
        budget = min(30.0, _time_left() - 5.0)
        sa_rows, sa_max = _sa_optimize(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Phase 5: Final local search polish
    if _time_left() > 5.0:
        budget = min(10.0, _time_left() - 2.0)
        ls_rows, ls_max = _local_search_reduce(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Verify
    if not _validate(best_rows):
        best_rows = _greedy_build(n_rows, k, rng)

    return best_rows
