"""Rewritten program for difference triangle set construction.

Uses a structured local-search / greedy hybrid:
  - Start from a greedy construction.
  - Improve via hill-climbing / simulated-annealing style moves on rows.
  - Each move replaces one row with a better one that reuses already-used
    differences less often, or shortens the max entry.
"""
import random
import time
import sys

# -----------------------------
# Helpers
# -----------------------------

def _row_diffs(row):
    """Return the set of positive differences within a row."""
    d = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(row[j] - row[i])
    return d


def _all_diffs(rows):
    """Return the multiset (or set) of all differences across rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _scope(rows):
    return max(max(row) for row in rows) if rows else 0


def _is_valid(rows):
    used = set()
    for row in rows:
        d = _row_diffs(row)
        if d & used:
            return False
        used |= d
    return True


def _greedy_row(k, used, max_val=None):
    """Greedily build a row of length k+1 starting at 0,
    using the smallest possible values whose differences are all in `used`
    (i.e., unused)."""
    row = [0]
    while len(row) < k + 1:
        found = False
        start = row[-1] + 1
        end = max_val if max_val is not None else 10**6
        for x in range(start, end + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            break
    if len(row) != k + 1:
        return None
    return row


def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(k, used)
        if row is None:
            return None
        rows.append(row)
    return rows


def _random_row(k, used, max_val):
    """Try to construct a random row that avoids `used` differences.
    Return None if impossible within max_val."""
    row = [0]
    while len(row) < k + 1:
        candidates = list(range(row[-1] + 1, max_val + 1))
        random.shuffle(candidates)
        found = False
        for x in candidates:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row


# -----------------------------
# Local search
# -----------------------------

def _local_search(rows, k, n_rows, used, max_val, time_limit, rng):
    """Attempt to improve rows via row-replacement moves."""
    rows = [list(r) for r in rows]
    best = rows[:]
    best_scope = _scope(best)
    t0 = time.time()

    # Precompute used set
    used = set()
    for r in rows:
        used |= _row_diffs(r)

    # We'll try to replace one row at a time with a better one.
    # A "better" row either has a smaller max value, or the same max but
    # allows overall scope reduction when combined.
    # For simplicity: try to find a row that, when substituted, yields
    # a valid full set with the same or smaller scope.

    # We track current scope.
    current_scope = _scope(rows)

    # We'll do random restarts of the whole construction plus local search.
    # For now, just do simple row swaps.

    while time.time() - t0 < time_limit:
        # Pick a random row to replace
        idx = rng.randrange(n_rows)
        old_row = rows[idx]
        old_diffs = _row_diffs(old_row)

        # Remove old row's diffs from used
        used -= old_diffs

        # Try to find a new row with max <= current_scope that is valid
        new_row = _random_row(k, used, current_scope)
        if new_row is not None:
            new_diffs = _row_diffs(new_row)
            if not (new_diffs & used):
                # Accept
                used |= new_diffs
                rows[idx] = new_row
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best = [list(r) for r in rows]
                    best_scope = new_scope
        else:
            # Restore
            used |= old_diffs

    return best, best_scope


# -----------------------------
# Main entry point
# -----------------------------

def construct(n_rows, k):
    """Construct a difference triangle set of n_rows rows, each of length k+1.
    Minimize the largest entry (scope)."""
    rng = random.Random(42)
    t_start = time.time()
    deadline = t_start + 140  # Leave margin before 150s kill

    # Initial greedy construction
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        # Fallback: random search
        rows = []
        used = set()
        for _ in range(n_rows):
            for _ in range(1000):
                r = _random_row(k, used, 500)
                if r is not None:
                    rows.append(r)
                    used |= _row_diffs(r)
                    break
            else:
                # Should not happen
                return None
        # Ensure validity
        if not _is_valid(rows):
            # Last resort
            return _greedy_construct(n_rows, k) or []
    else:
        # Ensure validity
        if not _is_valid(rows):
            return []

    best = [list(r) for r in rows]
    best_scope = _scope(best)

    # Local search
    remaining = deadline - time.time()
    if remaining > 5:
        improved, scope = _local_search(rows, k, n_rows, _all_diffs(rows),
                                        best_scope, remaining - 5, rng)
        if scope < best_scope:
            best = improved
            best_scope = scope

    # If we have time, try a fresh restart with different seed
    remaining = deadline - time.time()
    if remaining > 10:
        rng2 = random.Random(123)
        for seed in [7, 11, 13]:
            remaining = deadline - time.time()
            if remaining < 5:
                break
            rng2.seed(seed)
            rows2 = _greedy_construct(n_rows, k)
            if rows2 is not None and _is_valid(rows2):
                s2 = _scope(rows2)
                if s2 < best_scope:
                    best = rows2
                    best_scope = s2
                # Local search on this
                rem2 = deadline - time.time()
                if rem2 > 3:
                    improved2, scope2 = _local_search(rows2, k, n_rows,
                                                      _all_diffs(rows2),
                                                      best_scope, rem2 - 3, rng2)
                    if scope2 < best_scope:
                        best = improved2
                        best_scope = scope2

    return best
