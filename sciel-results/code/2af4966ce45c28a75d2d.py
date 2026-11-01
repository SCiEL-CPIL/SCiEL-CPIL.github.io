"""Rewritten program: randomized backtracking with M-incrementation and deadline guard."""
import time
import random

_DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _row_diffs(row):
    """Return the set of positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _greedy_row(used, k, rng, max_m=None):
    """
    Greedily build a row of k+1 increasing integers starting at 0,
    choosing the smallest available next value whose differences are unused.
    Uses randomization: occasionally skip the smallest available to explore alternatives.
    Returns the row or None if no valid row can be built within max_m (if given).
    """
    row = [0]
    used_local = set()  # differences used within this row
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_m is not None and x > max_m:
            return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used) and not (new_diffs & used_local):
            # With some probability, accept; otherwise try next x (randomized greedy)
            if rng.random() < 0.85:
                row.append(x)
                used_local |= new_diffs
            # else: skip this x, try next
    return row


def _backtrack_row(used, k, depth, max_m, rng):
    """
    Backtracking search for a single row given the global 'used' set of differences.
    Returns a valid row (list of k+1 ints) or None.
    """
    if depth == k + 1:
        return []  # base case: empty remaining
    row = [0]
    used_local = set()
    x = 0

    # We'll do a DFS: try each candidate x in increasing order
    # But to keep it tractable, limit branching
    best = None

    def dfs(pos, current_row, current_used_local, last_x):
        nonlocal best
        if best is not None:
            return True
        if pos == k + 1:
            best = current_row[:]
            return True

        # Try candidates starting from last_x + 1
        start = last_x + 1
        # Limit how far we search to keep runtime bounded
        end = min(max_m + 1, start + 40)  # try at most 40 candidates per level

        for x in range(start, end):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used or new_diffs & current_used_local:
                continue
            # Check within current row
            current_row.append(x)
            current_used_local |= new_diffs
            if dfs(pos + 1, current_row, current_used_local, x):
                current_row.pop()
                current_used_local -= new_diffs
                return True
            current_row.pop()
            current_used_local -= new_diffs
        return False

    dfs(1, [0], set(), 0)
    return best


def _construct_randomized(n_rows, k, rng, max_m, time_limit):
    """
    Attempt to construct n_rows rows with scope <= max_m using randomized backtracking.
    Returns list of rows or None.
    """
    start_time = time.time()
    used = set()
    rows = []

    for _ in range(n_rows):
        if time.time() - start_time > time_limit:
            return None
        # Try randomized greedy first (fast)
        row = None
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                return None
            row = _greedy_row(used, k, rng, max_m)
            if row is not None:
                break
        if row is None:
            # Fall back to backtracking
            row = _backtrack_row(used, k, 0, max_m, rng)
            if row is None:
                return None
        rows.append(row)
        used |= _row_diffs(row)

    return rows


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows) if rows else 0


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope), subject to all positive differences
    within rows being distinct across all rows.
    """
    global _DEADLINE
    start_time = time.time()
    deadline = start_time + _DEADLINE

    rng = random.Random(42)

    # Lower bound: counting argument
    # Each row of k+1 elements has C(k+1, 2) differences.
    # Total differences needed: n_rows * C(k+1, 2)
    # Max possible differences in [1, m]: m
    # So m >= n_rows * C(k+1, 2)
    total_diffs = n_rows * (k + 1) * k // 2
    lower_bound = total_diffs

    # Start searching from lower bound upward
    best_rows = None
    best_scope = float('inf')

    # First, try a pure greedy to get an initial upper bound
    for m_try in range(max(lower_bound, 50), 200):
        if time.time() > deadline - 5:
            break
        rows = _construct_randomized(n_rows, k, rng, m_try, 3.0)
        if rows is not None:
            scope = _scope(rows)
            if scope <= m_try:
                best_rows = rows
                best_scope = scope
                break

    if best_rows is None:
        # Fallback: pure greedy without scope limit
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, None)
            if row is None:
                # Very unlikely, but handle it
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
            best_rows.append(row)
        best_scope = _scope(best_rows)

    # Now try to improve: search for smaller scope
    # Binary search or linear search downward from best_scope
    search_start = best_scope
    search_end = max(lower_bound, best_scope - 20)

    for m_try in range(search_start - 1, search_end - 1, -1):
        if time.time() > deadline - 10:
            break
        # Multiple attempts with different random seeds
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            rng_attempt = random.Random(rng.random() * 1000000)
            rows = _construct_randomized(n_rows, k, rng_attempt, m_try, 10.0)
            if rows is not None:
                scope = _scope(rows)
                if scope <= m_try:
                    best_rows = rows
                    best_scope = scope
                    found = True
                    break
        if found:
            continue
        else:
            # If we failed at m_try, likely can't do better; stop
            # But try one more step down to be safe
            if m_try - search_end > 10:
                break

    return best_rows
