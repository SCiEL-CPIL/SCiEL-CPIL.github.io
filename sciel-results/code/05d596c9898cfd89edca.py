"""Difference triangle set (7,5) - local search / simulated annealing design."""
import random
import time

_DEADLINE = 145.0  # safety margin under 150s wall-clock


def _all_diffs(row):
    """Return set of positive differences within a single row."""
    n = len(row)
    s = set()
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s


def _validate(rows):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        for d in _all_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy(n_rows, k):
    """Greedy construction: smallest next mark with no new difference collisions."""
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


def _random_valid_start(n_rows, k, rng):
    """Build a valid starting solution with random tie-breaking for speed."""
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


def _row_diffs(row):
    n = len(row)
    s = set()
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s


def _local_search(rows, n_rows, k, deadline, rng):
    """Simulated annealing / hill-climbing to reduce scope while keeping validity."""
    rows = [list(r) for r in rows]
    best = rows
    best_scope = _scope(rows)

    # Precompute per-row diff sets and global used set
    row_diffs = [_row_diffs(r) for r in rows]
    used = set()
    for s in row_diffs:
        used |= s

    def try_decrease(r, j):
        """Try to decrease rows[r][j] by 1. Returns True if valid and applied."""
        nonlocal used
        row = rows[r]
        old = row[j]
        new_val = old - 1
        # must stay strictly increasing: new_val > row[j-1]
        if new_val <= row[j - 1]:
            return False
        # remove old diffs involving position j
        old_diffs = set()
        for i in range(j):
            old_diffs.add(old - row[i])
        for i in range(j + 1, len(row)):
            old_diffs.add(row[i] - old)
        # check new diffs don't collide with (used - old_diffs)
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - row[i])
        for i in range(j + 1, len(row)):
            new_diffs.add(row[i] - new_val)
        # new diffs must not collide with other rows' diffs
        others = used - old_diffs
        if new_diffs & others:
            return False
        # also new_diffs internally must be distinct (they are, since row is strictly increasing)
        # apply
        row[j] = new_val
        used = (used - old_diffs) | new_diffs
        row_diffs[r] = new_diffs | (row_diffs[r] - old_diffs)
        # recompute row_diffs[r] properly
        row_diffs[r] = _row_diffs(row)
        return True

    temperature = best_scope * 0.5
    cooling = 0.999
    min_temp = 1e-3

    while time.time() < deadline and temperature > min_temp:
        r = rng.randrange(n_rows)
        j = rng.randrange(1, k + 1)
        row = rows[r]
        if j >= len(row) or row[j] <= row[j - 1]:
            temperature *= cooling
            continue
        old_scope = _scope(rows)
        # attempt decrease
        # save state
        saved_row = list(row)
        saved_used = used.copy()
        saved_row_diffs_r = row_diffs[r].copy()
        if try_decrease(r, j):
            new_scope = _scope(rows)
            delta = new_scope - old_scope
            if delta <= 0:
                # improvement or equal; keep
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(x) for x in rows]
            else:
                # worse; accept with prob
                if rng.random() < _boltzmann(-delta, temperature):
                    pass  # keep
                else:
                    # revert
                    rows[r] = saved_row
                    used = saved_used
                    row_diffs[r] = saved_row_diffs_r
        temperature *= cooling

    return best


def _boltzmann(delta, T):
    if T <= 0:
        return 0.0
    import math
    if delta / T > 50:
        return 0.0
    return math.exp(-delta / T)


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    # Initial valid solution via greedy
    rows = _greedy(n_rows, k)
    best = [list(r) for r in rows]
    best_scope = _scope(best)

    # Local search to reduce scope
    if time.time() < deadline - 5:
        improved = _local_search(rows, n_rows, k, deadline, rng)
        if _scope(improved) < best_scope:
            best = improved
            best_scope = _scope(improved)

    # Final validation
    if not _validate(best):
        # fall back to greedy
        best = _greedy(n_rows, k)

    return best
