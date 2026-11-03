"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _backtrack_row(used, k, rng, max_depth, deadline):
    """Backtrack to build a row of k+1 elements starting at 0,
    avoiding differences in 'used'. Returns row or None."""
    row = [0]
    
    def bt(depth, current_max):
        if time.time() > deadline:
            return False
        if depth == k:
            return True
        # Try values from current_max+1 upward, with some randomization
        start = current_max + 1
        # Randomize the order of trying values to escape local traps
        # Use a window: try a few values in random order, then continue sequentially
        window_size = min(15, max(5, (k - depth) * 3))
        candidates = list(range(start, start + window_size))
        rng.shuffle(candidates)
        # Also add some larger candidates
        for extra in range(window_size, window_size + 5):
            candidates.append(start + extra)
        
        tried = set()
        for x in candidates:
            if x in tried:
                continue
            tried.add(x)
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used.update(new_diffs)
            if bt(depth + 1, x):
                return True
            used.difference_update(new_diffs)
            row.pop()
            if len(tried) > max_depth:
                break
        # If window exhausted, try sequentially beyond
        for x in range(start + window_size + 5, start + window_size + 50):
            if x in tried:
                continue
            tried.add(x)
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used.update(new_diffs)
            if bt(depth + 1, x):
                return True
            used.difference_update(new_diffs)
            row.pop()
            if len(tried) > max_depth * 2:
                break
        return False
    
    if bt(1, 0):
        return row
    return None

def _greedy_row(used, k):
    """Build one row greedily given the set of already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used.update(new_diffs)
    return row

def _try_compact_row(row, other_diffs, k):
    """Try to compact a single row: pull each element left to smallest valid position.
    Returns (new_row, success)."""
    new_row = [0]
    for pos in range(1, len(row)):
        orig = row[pos]
        x = new_row[-1] + 1
        placed = False
        while x <= orig:
            new_diffs = {x - a for a in new_row}
            if not new_diffs & other_diffs:
                new_row.append(x)
                placed = True
                break
            x += 1
        if not placed:
            return row, False
    return new_row, True

def _local_search(rows, used, n_rows, k, rng, deadline):
    """Local search: try to compact individual rows."""
    cur_rows = [list(r) for r in rows]
    cur_used = set()
    for r in cur_rows:
        cur_used.update(_row_diffs(r))
    cur_scope = _scope(cur_rows)
    
    T = 30.0
    T_min = 0.5
    cooling = 0.998
    
    while time.time() < deadline and T > T_min:
        ri = rng.randrange(n_rows)
        row = cur_rows[ri]
        
        other_diffs = set()
        for idx, r in enumerate(cur_rows):
            if idx == ri:
                continue
            other_diffs.update(_row_diffs(r))
        
        new_row, ok = _try_compact_row(row, other_diffs, k)
        if ok and new_row != row:
            new_scope = max(max(r) for r in cur_rows[:ri] + [new_row] + cur_rows[ri+1:])
            all_d = set(other_diffs)
            all_d.update(_row_diffs(new_row))
            expected = n_rows * k * (k + 1) // 2
            if len(all_d) == expected:
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < min(1.0, (-delta / T) if delta > 0 else 1.0):
                    cur_rows[ri] = new_row
                    cur_scope = new_scope
                    cur_used = set()
                    for r in cur_rows:
                        cur_used.update(_row_diffs(r))
        
        T *= cooling
    
    return cur_rows, cur_scope

def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple randomized backtracking constructions
    num_attempts = 0
    while time.time() < deadline - 20 and num_attempts < 50:
        num_attempts += 1
        used = set()
        rows = []
        success = True
        
        for r in range(n_rows):
            if time.time() > deadline - 20:
                success = False
                break
            # Use backtracking with some randomization
            row = _backtrack_row(used, k, rng, max_depth=30, deadline=deadline - 20)
            if row is None:
                # Fall back to greedy
                row = _greedy_row(used, k)
            rows.append(row)
            used.update(_row_diffs(row))
        
        if not success or len(rows) != n_rows:
            continue
        
        # Verify validity
        all_d = set()
        for r in rows:
            all_d.update(_row_diffs(r))
        expected = n_rows * k * (k + 1) // 2
        if len(all_d) != expected:
            continue
        
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        
        # If we've found a very good solution, do local search
        if scope <= 135 and time.time() < deadline - 10:
            ls_rows, ls_scope = _local_search(rows, used, n_rows, k, rng, deadline - 5)
            if ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = [list(r) for r in ls_rows]
        
        # If we hit a good bound, try more attempts with different seeds
        if best_scope <= 125:
            break
    
    # Phase 2: Local search on best solution
    if best_rows is not None and time.time() < deadline - 5:
        used = set()
        for r in best_rows:
            used.update(_row_diffs(r))
        ls_rows, ls_scope = _local_search(best_rows, used, n_rows, k, rng, deadline - 2)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [list(r) for r in ls_rows]
    
    # Final compaction
    if best_rows is not None:
        for pass_num in range(3):
            if time.time() > deadline - 2:
                break
            for ri in range(n_rows):
                other_diffs = set()
                for idx, r in enumerate(best_rows):
                    if idx == ri:
                        continue
                    other_diffs.update(_row_diffs(r))
                new_row, ok = _try_compact_row(best_rows[ri], other_diffs, k)
                if ok and new_row != best_rows[ri]:
                    all_d = set(other_diffs)
                    all_d.update(_row_diffs(new_row))
                    expected = n_rows * k * (k + 1) // 2
                    if len(all_d) == expected:
                        best_rows[ri] = new_row
                        best_scope = _scope(best_rows)
    
    if best_rows is None:
        # Emergency fallback: pure greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            best_rows.append(_greedy_row(used, k))
            used.update(_row_diffs(best_rows[-1]))
    
    return best_rows
# EVOLVE-BLOCK-END
