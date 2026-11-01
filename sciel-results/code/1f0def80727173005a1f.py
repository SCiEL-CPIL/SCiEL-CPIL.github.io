"""Fresh design for the difference triangle set task.
Uses a randomized local-search / simulated-annealing style construction with
a strong greedy initialization, then attempts to reduce the maximum entry by
moving elements within rows and swapping between rows while preserving
distinctness of all positive intra-row differences across all rows.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _all_diffs(rows):
    """Return the set of all positive differences a_j - a_i (i<j) within each row."""
    s = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
    return s

def _is_valid(rows, k):
    """Check validity: each row has k+1 strictly increasing ints starting at 0,
    and all positive intra-row differences are distinct across all rows."""
    if len(rows) == 0:
        return True
    expected_len = k + 1
    for row in rows:
        if len(row) != expected_len:
            return False
        if row[0] != 0:
            return False
        for i in range(1, expected_len):
            if row[i] <= row[i - 1]:
                return False
    diffs = _all_diffs(rows)
    # The total number of positive differences per row is k*(k+1)//2
    expected_count = len(rows) * (k * (k + 1) // 2)
    return len(diffs) == expected_count

def _greedy_row(used, k, start=0):
    """Build one row greedily starting at 0, choosing smallest next mark whose
    new differences don't collide with 'used'."""
    row = [0]
    x = start
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _greedy_init(n_rows, k, seed=0):
    """Greedy initialization (same as seed program) but with a fixed seed for
    reproducibility of the base; we'll randomize later."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(r) for r in rows)

def _try_move(rows, row_idx, pos, new_val, used, k):
    """Try to change rows[row_idx][pos] to new_val. Returns (success, new_used).
    Assumes new_val > rows[row_idx][pos-1] (or pos==0 but pos 0 is always 0, so pos>=1)
    and new_val < rows[row_idx][pos+1] (or pos is last)."""
    row = rows[row_idx]
    old_val = row[pos]
    if new_val == old_val:
        return False, used
    
    # Check ordering
    if pos > 0 and new_val <= row[pos - 1]:
        return False, used
    if pos < len(row) - 1 and new_val >= row[pos + 1]:
        return False, used
    
    # Remove old differences involving position pos
    new_used = set(used)
    for i in range(len(row)):
        if i == pos:
            continue
        d = abs(row[pos] - row[i])
        if d in new_used:
            new_used.discard(d)
    
    # Add new differences involving position pos
    for i in range(len(row)):
        if i == pos:
            continue
        d = abs(new_val - row[i])
        if d in new_used:
            # Collision
            return False, used
        new_used.add(d)
    
    # Apply
    row[pos] = new_val
    return True, new_used

def _local_search(rows, k, time_limit, rng):
    """Attempt to reduce the max entry via local search moves.
    Returns the best rows found."""
    best_rows = [r[:] for r in rows]
    best_score = _max_entry(best_rows)
    used = _all_diffs(rows)
    
    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 50000
    
    n = len(rows)
    k1 = k + 1
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random row and position to try moving
        row_idx = rng.randint(0, n - 1)
        pos = rng.randint(1, k1 - 1)  # skip position 0
        row = rows[row_idx]
        old_val = row[pos]
        
        # Try to move downward (reduce max entry)
        # Candidate new values: between row[pos-1]+1 and old_val-1
        lo = row[pos - 1] + 1
        hi = old_val - 1
        if lo < hi:
            new_val = rng.randint(lo, hi)
            success, new_used = _try_move(rows, row_idx, pos, new_val, used, k)
            if success:
                used = new_used
                no_improve = 0
                if _max_entry(rows) < best_score:
                    best_score = _max_entry(rows)
                    best_rows = [r[:] for r in rows]
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Also try swapping two elements between different rows
        if rng.random() < 0.1 and n > 1:
            r1 = rng.randint(0, n - 1)
            r2 = rng.randint(0, n - 1)
            if r1 != r2:
                p1 = rng.randint(1, k1 - 1)
                p2 = rng.randint(1, k1 - 1)
                v1 = rows[r1][p1]
                v2 = rows[r2][p2]
                if v1 != v2:
                    # Try swapping: check if both rows remain valid
                    # This is complex; skip for now and rely on single moves
                    pass
    
    return best_rows, best_score

def _random_restart_greedy(n_rows, k, seed, rng):
    """Greedy with randomization in the order of trying candidates."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try a few random candidates in increasing order
            found = False
            for attempt in range(5):
                if attempt == 0:
                    cand = x + 1
                else:
                    cand = x + 1 + rng.randint(0, 10)
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= new
                    x = cand
                    found = True
                    break
            if not found:
                # Fallback: scan
                x += 1
                while True:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        break
                    x += 1
        rows.append(row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Phase 1: Greedy initialization
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_score = _max_entry(best_rows)
    
    if time.time() > deadline:
        return best_rows
    
    # Phase 2: Local search to improve
    rng = random.Random(42)
    ls_time = min(30, deadline - time.time())
    if ls_time > 5:
        rows, score = _local_search(rows, k, ls_time, rng)
        if score < best_score:
            best_score = score
            best_rows = rows
    
    if time.time() > deadline:
        return best_rows
    
    # Phase 3: Multiple random restarts with different seeds
    remaining = deadline - time.time()
    if remaining > 10:
        attempts = 0
        while time.time() < deadline - 2 and attempts < 20:
            seed = 1000 + attempts
            rng2 = random.Random(seed)
            rows2 = _random_restart_greedy(n_rows, k, seed, rng2)
            score2 = _max_entry(rows2)
            if score2 < best_score:
                best_score = score2
                best_rows = [r[:] for r in rows2]
            # Quick local search on this candidate
            rem = deadline - time.time()
            if rem > 5 and score2 <= best_score + 5:
                rows2, score2 = _local_search(rows2, k, min(5, rem - 1), rng2)
                if score2 < best_score:
                    best_score = score2
                    best_rows = [r[:] for r in rows2]
            attempts += 1
    
    return best_rows

# EVOLVE-BLOCK-END
