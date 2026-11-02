"""Difference triangle set (7,5): minimize the largest entry (scope)."""
import time
import random
from itertools import combinations

# Deadline guard: finish well before the 120s kill (use ~110s budget for search)
_DEADLINE = 110.0


def _row_diffs(row):
    """Return the set of all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    all_d = set()
    for r in rows:
        all_d |= _row_diffs(r)
    return all_d


def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    all_d = set()
    for r in rows:
        rd = _row_diffs(r)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, start_hint=1):
    """
    Greedily build one row of k+1 increasing integers starting at 0.
    Each new mark is the smallest x > last such that all new differences
    to earlier marks in this row are not in 'used'.
    Returns (row, new_diffs).
    """
    row = [0]
    new_diffs = set()
    x = start_hint
    while len(row) < k + 1:
        # Try x, x+1, ... until we find a valid mark
        found = False
        while True:
            cand_diffs = {x - a for a in row}
            if not (cand_diffs & used) and not (cand_diffs & new_diffs):
                row.append(x)
                new_diffs |= cand_diffs
                found = True
                break
            x += 1
        x = row[-1] + 1  # next candidate must be > last mark
    return row, new_diffs


def _greedy_rows(n_rows, k, order=None):
    """
    Build n_rows rows greedily. 'order' is a permutation of row indices
    to control which row is built in which slot (doesn't change final set,
    but can affect greedy choices if we later add randomization).
    Returns list of rows.
    """
    used = set()
    rows = [None] * n_rows
    for idx in (order if order is not None else range(n_rows)):
        row, nd = _greedy_row(used, k)
        rows[idx] = row
        used |= nd
    return rows


def _backtrack_row(used, k, time_limit=None, start_time=None):
    """
    Backtracking search for a single row of k+1 marks starting at 0,
    minimizing the largest mark, subject to no new difference colliding
    with 'used'.
    Returns (row, new_diffs) or None if not found within time.
    """
    if start_time is None:
        start_time = time.time()
    
    row = [0]
    new_diffs = set()
    
    def search(last_mark, target_len):
        if time.time() - start_time > time_limit:
            return False
        if len(row) == target_len:
            return True
        # Try next marks starting from last_mark + 1
        x = last_mark + 1
        while True:
            cand = {x - a for a in row}
            if not (cand & used) and not (cand & new_diffs):
                row.append(x)
                new_diffs |= cand
                if search(x, target_len):
                    return True
                row.pop()
                new_diffs -= cand
            x += 1
            # Safety: don't go too far
            if x > last_mark + 200:
                return False
        return False
    
    if search(0, k + 1):
        return list(row), set(new_diffs)
    return None


def _local_search(rows, n_rows, k, time_limit, start_time):
    """
    Improve a set of rows by trying to rebuild individual rows with
    backtracking to find a row with a smaller maximum mark.
    Returns improved rows or original if no improvement.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Compute used differences excluding each row in turn
    all_diffs = _all_diffs(best_rows)
    
    for attempt in range(n_rows * 3):  # Try multiple times
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random row to try to improve
        idx = random.randrange(n_rows)
        row_diffs = _row_diffs(best_rows[idx])
        other_diffs = all_diffs - row_diffs
        
        # Try to find a better row (smaller max) using backtracking
        result = _backtrack_row(other_diffs, k, time_limit=1.0, start_time=time.time())
        if result is None:
            continue
        new_row, new_diffs = result
        if max(new_row) < max(best_rows[idx]):
            # Validate
            all_d = set()
            valid = True
            for i, r in enumerate(best_rows):
                if i == idx:
                    rd = new_diffs
                else:
                    rd = _row_diffs(r)
                if rd & all_d:
                    valid = False
                    break
                all_d |= rd
            if valid:
                best_rows[idx] = new_row
                all_diffs = all_d
                best_scope = _scope(best_rows)
    
    return best_rows


def _randomized_greedy(n_rows, k, seed, time_limit, start_time):
    """
    Randomized greedy: build rows in random order, with some randomness
    in mark selection (try a few candidates and pick randomly among valid).
    """
    rng = random.Random(seed)
    used = set()
    rows = [None] * n_rows
    order = list(range(n_rows))
    rng.shuffle(order)
    
    for idx in order:
        if time.time() - start_time > time_limit:
            break
        row = [0]
        new_diffs = set()
        x = 1
        while len(row) < k + 1:
            # Try a few candidates, pick the first valid (greedy) or
            # occasionally skip to a slightly larger one for diversity
            found = False
            tries = 0
            while not found and tries < 50:
                cand = {x - a for a in row}
                if not (cand & used) and not (cand & new_diffs):
                    # With small probability, skip this and try x+1
                    if rng.random() < 0.05 and x + 1 > row[-1]:
                        x += 1
                        tries += 1
                        continue
                    row.append(x)
                    new_diffs |= cand
                    found = True
                else:
                    x += 1
                    tries += 1
            if not found:
                # Fallback: pure greedy
                x = row[-1] + 1
                while True:
                    cand = {x - a for a in row}
                    if not (cand & used) and not (cand & new_diffs):
                        row.append(x)
                        new_diffs |= cand
                        break
                    x += 1
            x = row[-1] + 1
        rows[idx] = row
        used |= new_diffs
    
    # Fill any missing rows with greedy
    for i in range(n_rows):
        if rows[i] is None:
            if time.time() - start_time > time_limit:
                break
            row, nd = _greedy_row(used, k)
            rows[i] = row
            used |= nd
    
    return rows


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Phase 1: Pure greedy baseline
    rows = _greedy_rows(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Phase 2: Local search to improve individual rows
    if time.time() < deadline - 5:
        improved = _local_search(best_rows, n_rows, k, time_limit=10.0, start_time=time.time())
        if _scope(improved) < best_scope:
            best_rows = improved
            best_scope = _scope(best_rows)
    
    # Phase 3: Randomized greedy with multiple seeds
    seed = 42
    while time.time() < deadline - 10:
        try:
            cand_rows = _randomized_greedy(n_rows, k, seed, time_limit=5.0, start_time=time.time())
            if _is_valid(cand_rows):
                s = _scope(cand_rows)
                if s < best_scope:
                    best_rows = [list(r) for r in cand_rows]
                    best_scope = s
        except Exception:
            pass
        seed += 1
    
    # Phase 4: Final local search on best
    if time.time() < deadline - 3:
        improved = _local_search(best_rows, n_rows, k, time_limit=3.0, start_time=time.time())
        if _scope(improved) < best_scope:
            best_rows = improved
            best_scope = _scope(improved)
    
    # Ensure validity
    if not _is_valid(best_rows):
        # Fallback to pure greedy
        best_rows = _greedy_rows(n_rows, k)
    
    return best_rows
