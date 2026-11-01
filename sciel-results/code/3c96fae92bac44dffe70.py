"""Structural rewrite: systematic backtracking with M-incrementation,
row-level re-solving, and randomized restarts.
The key idea is to treat each row as a unit and search for the smallest
M such that we can place n_rows rows of k+1 elements in [0, M] with
all pairwise differences distinct across all rows."""

import time
import random
import sys

def _valid(rows, n_rows, k):
    """Validate the full solution."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Return the set of all positive differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _build_greedy(n_rows, k):
    """Standard greedy construction: build each row one at a time,
    picking the smallest next value that doesn't conflict."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _backtrack_row(used, k, M, rng, depth_limit=200):
    """Try to construct a single row of k+1 elements starting at 0
    within [0, M] such that all its differences avoid 'used'.
    Returns (row, updated_used) or (None, None) on failure.
    Uses randomized backtracking."""
    row = [0]
    # We need to place k more elements in [1, M]
    # Use backtracking with randomization
    
    def bt(pos, row_so_far, used_so_far, depth):
        if depth > depth_limit:
            return None
        if pos == k + 1:
            return row_so_far[:]
        
        # Determine range of candidates for this position
        prev = row_so_far[-1]
        # Must be > prev and <= M
        # Also, the differences to all previous elements must not be in used_so_far
        # And the differences must be distinct within the row (they are if row is strictly increasing)
        
        # Collect valid candidates
        candidates = []
        for val in range(prev + 1, M + 1):
            # Check differences with all existing elements in this row
            valid = True
            for a in row_so_far:
                d = val - a
                if d in used_so_far:
                    valid = False
                    break
            if valid:
                candidates.append(val)
        
        if not candidates:
            return None
        
        # Randomize order for diversity
        rng.shuffle(candidates)
        
        # Try each candidate
        for val in candidates:
            new_diffs = set(val - a for a in row_so_far)
            new_used = used_so_far | new_diffs
            result = bt(pos + 1, row_so_far + [val], new_used, depth + 1)
            if result is not None:
                return result
        return None
    
    result = bt(1, row, used, 0)
    if result is None:
        return None, None
    # Compute final used set for this row
    final_diffs = _row_diffs_set(result)
    return result, used | final_diffs

def _try_construct_at_M(n_rows, k, M, rng, max_restarts=50):
    """Try to construct n_rows rows all within [0, M] using backtracking.
    Returns (rows, used) on success or (None, None) on failure."""
    for attempt in range(max_restarts):
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row, new_used = _backtrack_row(used, k, M, rng)
            if row is None:
                success = False
                break
            rows.append(row)
            used = new_used
        if success:
            return rows, used
    return None, None

def _resolve_row_given_others(rows, ri, n_rows, k, M, rng):
    """Re-solve row ri given all other rows are fixed.
    Returns (new_row, updated_used) or (None, None)."""
    # Compute used differences from all other rows
    used = set()
    for i in range(n_rows):
        if i == ri:
            continue
        used |= _row_diffs_set(rows[i])
    
    row, new_used = _backtrack_row(used, k, M, rng)
    if row is None:
        return None, None
    return row, used | _row_diffs_set(row)

def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin before 150s limit
    rng = random.Random(42)
    
    # Phase 1: Greedy to get a starting point
    best = _build_greedy(n_rows, k)
    best_val = _max_entry(best)
    
    # Phase 2: Try to improve via backtracking at lower M values
    # Start from the counting bound or slightly above
    # Counting bound: n_rows * k*(k+1)/2 differences must fit in [1, M]
    # So M >= n_rows * k*(k+1)/2
    counting_bound = n_rows * k * (k + 1) // 2
    
    # Try M values from max(counting_bound, k) up to best_val - 1
    start_M = max(counting_bound, k)
    
    # First, try to improve from the greedy solution downward
    for M in range(best_val - 1, start_M - 1, -1):
        if time.time() > deadline - 10:
            break
        rows, used = _try_construct_at_M(n_rows, k, M, rng, max_restarts=20)
        if rows is not None:
            if _valid(rows, n_rows, k):
                best = rows
                best_val = M
                break
    
    # Phase 3: If we found a good M, try to push further down
    if best_val > start_M:
        for M in range(best_val - 1, start_M - 1, -1):
            if time.time() > deadline - 10:
                break
            rows, used = _try_construct_at_M(n_rows, k, M, rng, max_restarts=30)
            if rows is not None:
                if _valid(rows, n_rows, k):
                    best = rows
                    best_val = M
    
    # Phase 4: Local improvement via row re-solving
    # Try to reduce the max by re-solving individual rows
    cur = [r[:] for r in best]
    cur_val = best_val
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        target = cur_val - 1
        if target < k:
            break
        for ri in range(n_rows):
            if time.time() > deadline - 5:
                break
            # Try to re-solve row ri within [0, target]
            new_row, _ = _resolve_row_given_others(cur, ri, n_rows, k, target, rng)
            if new_row is not None:
                # Verify the full solution
                cur[ri] = new_row
                if _valid(cur, n_rows, k):
                    new_max = _max_entry(cur)
                    if new_max < cur_val:
                        cur_val = new_max
                        best = [r[:] for r in cur]
                        improved = True
                    else:
                        # Revert if it didn't help
                        cur[ri] = best[ri][:]
                else:
                    cur[ri] = best[ri][:]
    
    # Phase 5: Randomized restarts with backtracking for further improvement
    while time.time() < deadline - 5:
        # Try random M values below current best
        target_M = rng.randint(start_M, max(start_M, best_val - 1))
        if target_M >= best_val:
            continue
        rows, used = _try_construct_at_M(n_rows, k, target_M, rng, max_restarts=10)
        if rows is not None and _valid(rows, n_rows, k):
            val = _max_entry(rows)
            if val < best_val:
                best = rows
                best_val = val
    
    # Final verification
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best
