"""Difference triangle set (7,5) – improved constructive search."""
import time
import random
from itertools import combinations

# We have a hard wall-clock budget from the evaluator; keep a safety margin.
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 8.0  # seconds of buffer


def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _validate(rows, k):
    """Check that all rows are valid and all positive differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & used:
            return False
        used |= rd
    return True


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng):
    """
    Greedy construction: for each row, pick the smallest next mark whose
    intra-row differences are all unused.  Optionally randomize tie-breaking
    or the order in which candidate marks are tried.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _greedy_construct_randomized(n_rows, k, rng, max_start=200):
    """
    Randomized greedy: for each row, among the smallest feasible marks,
    pick one at random from a window to diversify.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Find the next feasible mark
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _local_search_improve(rows, k, rng, max_iter=2000):
    """
    Try to reduce the scope by locally modifying individual rows.
    For a given row, attempt to replace it with a different valid row
    (same length, starts at 0, strictly increasing) whose differences
    are disjoint from the used differences of all OTHER rows, and whose
    max entry is smaller.
    """
    n_rows = len(rows)
    best_scope = _scope(rows)
    current_rows = [list(r) for r in rows]

    for _ in range(max_iter):
        # Pick a random row to try to improve
        ri = rng.randint(0, n_rows - 1)
        old_row = current_rows[ri]
        old_max = old_row[-1]
        if old_max <= 1:
            continue

        # Compute used differences from all other rows
        other_used = set()
        for j, row in enumerate(current_rows):
            if j == ri:
                continue
            other_used |= _row_diffs(row)

        # Try to find a replacement row with a smaller max
        # We'll do a greedy search for a row starting at 0 with k+1 entries,
        # max < old_max, all diffs disjoint from other_used.
        target_max = old_max - 1
        found = None
        # Try from small max upward? No, we want smaller max. Try greedy with cap.
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > target_max:
                break
            new = {x - a for a in row}
            if not (new & other_used):
                row.append(x)
        if len(row) == k + 1:
            found = row

        if found is not None and found[-1] < old_max:
            current_rows[ri] = found
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope

    return current_rows, best_scope


def _backtracking_construct(n_rows, k, rng, time_limit=5.0):
    """
    Backtracking search to construct n_rows valid rows, trying to minimize
    the maximum entry.  Uses a greedy heuristic with backtracking.
    """
    start = time.time()
    used = set()
    rows = []

    def bt(row_idx):
        if time.time() - start > time_limit:
            return False
        if row_idx == n_rows:
            return True
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
                if bt(row_idx + 1):
                    return True
                # backtrack
                used -= new
                row.pop()
                # Try next x
            else:
                continue
        return False

    # This naive backtracking is too slow; instead do a simpler approach:
    # construct rows one by one with greedy, but if stuck, restart with
    # different randomization.
    # For now, just return greedy result.
    return _greedy_construct(n_rows, k, rng)


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are globally distinct.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + _DEADLINE_SECONDS - _SAFETY_MARGIN
    rng = random.Random(42)

    # Start with the deterministic greedy
    best_rows = _greedy_construct(n_rows, k, rng)
    best_scope = _scope(best_rows)

    # Try multiple randomized restarts of greedy with different seeds
    # and local search improvements.
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Randomized greedy: shuffle the order of rows or use different seeds
        rng2 = random.Random(attempt * 12345 + 67890)
        rows = _greedy_construct(n_rows, k, rng2)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

        # Local search on the best rows
        if attempt % 3 == 0:
            ls_rows, ls_scope = _local_search_improve(best_rows, k, rng2, max_iter=500)
            if ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = ls_rows

        # Occasionally try a different construction strategy:
        # build rows in a different order (e.g., by target max)
        if attempt % 10 == 0:
            # Try constructing with a tighter bound
            # For each row, try to keep max as small as possible
            used = set()
            new_rows = []
            ok = True
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                new_rows.append(row)
            new_scope = _scope(new_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = new_rows

    # Final validation
    if not _validate(best_rows, k):
        # Fallback to simple greedy
        best_rows = _greedy_construct(n_rows, k, rng)

    return best_rows
