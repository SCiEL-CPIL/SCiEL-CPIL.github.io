import time
import random
import sys

# EVOLVE-BLOCK-START
def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _scope(rows):
    return max(max(r) for r in rows)

def _valid(rows):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True

def _greedy_row(used, k, rng, jitter=0, max_scan=None):
    """Build one row of k+1 entries starting at 0, using randomized greedy.
    
    At each step, find the smallest valid x, then possibly advance by a random
    amount up to jitter to explore alternatives.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        # Find the smallest valid x
        found = False
        scan_limit = max_scan if max_scan else x + 500
        while x < scan_limit:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                # Found a valid x; apply jitter
                if jitter > 0:
                    # With some probability, skip ahead a bit
                    skip = rng.randint(0, jitter)
                    # Try to advance by skip while staying valid
                    for _ in range(skip):
                        x += 1
                        new2 = {x - a for a in row}
                        if new2 & used:
                            break
                        new = new2
                row.append(x)
                used |= new
                found = True
                break
        if not found:
            # Fallback: extend scan limit
            scan_limit = x + 2000
            while x < scan_limit:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    found = True
                    break
            if not found:
                return None
    return row

def _build_solution(n_rows, k, rng, jitter=2, max_scan=None):
    """Build a complete solution with randomized greedy. Returns (rows, scope) or (None, inf)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, jitter=jitter, max_scan=max_scan)
        if row is None:
            return None, float('inf')
        rows.append(row)
    return rows, _scope(rows)

def _try_improve_last_mark(rows, k, rng, used, row_idx, max_trials=200):
    """Try to reduce the last mark of a given row by re-filling from a smaller start."""
    row = rows[row_idx]
    if len(row) < k + 1:
        return False
    # Remove the last mark
    last = row[-1]
    # Compute differences that involved the last mark
    last_diffs = {last - a for a in row[:-1]}
    # Remove from used
    used -= last_diffs
    row.pop()
    
    best_new = last
    # Try to find a smaller valid last mark
    # The last mark must be > row[-1] (the new last element after pop)
    start = row[-1] + 1
    # Try values from start upward, but we want to go DOWN from last
    # Actually we want to try values < last that are still > row[-1]
    candidates = list(range(row[-1] + 1, last))
    rng.shuffle(candidates)
    # Also try a few values near last
    for c in candidates[:max_trials]:
        new = {c - a for a in row}
        if not (new & used):
            best_new = min(best_new, c)
    # Try to place at best_new
    if best_new < last:
        new = {best_new - a for a in row}
        if not (new & used):
            row.append(best_new)
            used |= new
            return True
    else:
        # Restore original
        row.append(last)
        used |= last_diffs
    return False

def _local_search(rows, k, used, rng, time_limit, max_passes=5):
    """Try to reduce scope via local moves."""
    start_time = time.time()
    for _ in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        # Find the row with the largest last element
        max_row = max(range(len(rows)), key=lambda i: rows[i][-1])
        improved = _try_improve_last_mark(rows, k, rng, used, max_row, max_trials=500)
        if not improved:
            # Try a random row
            ridx = rng.randint(0, len(rows) - 1)
            _try_improve_last_mark(rows, k, rng, used, ridx, max_trials=300)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Deterministic seed for reproducibility but allow some randomness
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Pure greedy baseline (fast, gives a valid solution immediately)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, jitter=0)
        if row is None:
            # Should not happen for reasonable inputs
            break
        rows.append(row)
    if len(rows) == n_rows:
        best_rows = [r[:] for r in rows]
        best_scope = _scope(rows)
    
    # Phase 2: Randomized greedy restarts
    attempt = 0
    while time.time() - start_time < deadline - 10:
        attempt += 1
        # Vary jitter and seed
        rng2 = random.Random(42 + attempt * 7919)
        jitter = rng2.choice([0, 1, 2, 3, 5, 8, 12])
        rows2, scope2 = _build_solution(n_rows, k, rng2, jitter=jitter)
        if rows2 is not None and scope2 < best_scope:
            # Verify
            if _valid(rows2):
                best_rows = [r[:] for r in rows2]
                best_scope = scope2
    
    # Phase 3: Local search on the best solution
    if best_rows is not None and time.time() - start_time < deadline - 5:
        used_best = set()
        for row in best_rows:
            used_best |= _row_diffs(row)
        rng3 = random.Random(12345)
        local_time = min(30, deadline - 5 - (time.time() - start_time))
        if local_time > 2:
            best_rows = _local_search(best_rows, k, used_best, rng3, local_time, max_passes=10)
            new_scope = _scope(best_rows)
            if _valid(best_rows) and new_scope < best_scope:
                best_scope = new_scope
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, jitter=0)
            rows.append(row)
        return rows
    
    return best_rows
# EVOLVE-BLOCK-END
