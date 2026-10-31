"""Difference triangle set (4,7): minimize the largest entry m over n_rows rows of k+1
increasing integers starting at 0, with all positive within-row differences distinct
across every row.

Design (within the greedy + SA + local-polish family, not a new family):
  1. Greedy ruler construction for a quick valid baseline.
  2. Simulated annealing with fast integer conflict-counting that optimizes the
     *binding* (largest) row first, escaping greedy local minima.
  3. Per-row local polish (single-mark moves) to refine the result.
  4. Explicit wall-clock deadline guard returns the best valid result found.
"""
import time
import random
import math

_DEADLINE = 150.0


def _now():
    return time.time()


def _deadline_reached(t0, margin=2.0):
    return (_now() - t0) > (_DEADLINE - margin)


def _row_diffs_set(row):
    """Set of positive differences a_j - a_i (i<j) for one row."""
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s


def _all_diffs(rows):
    """Union of all within-row positive differences across rows."""
    used = set()
    for row in rows:
        used |= _row_diffs_set(row)
    return used


def _conflict_count(rows):
    """Number of difference values used more than once (each extra copy counts).
    A solution is valid iff this is 0."""
    cnt = {}
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                cnt[d] = cnt.get(d, 0) + 1
    conf = 0
    for v in cnt.values():
        if v > 1:
            conf += v - 1
    return conf


def _greedy_construct(n_rows, k):
    """Each row starts at 0 and takes the smallest next mark whose differences to the
    earlier marks of the row are all still unused by any row."""
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


def _local_polish(rows, k, t0, iters):
    """Try single-mark moves (change row[r][pos] to a new value keeping the row
    strictly increasing) that reduce total conflicts; accept equal-conflict moves
    that reduce the largest entry. Returns improved rows."""
    rows = [list(r) for r in rows]
    best_conf = _conflict_count(rows)
    best_m = max(max(r) for r in rows)
    for _ in range(iters):
        if _deadline_reached(t0):
            break
        r = random.randrange(len(rows))
        pos = random.randrange(1, k + 1)  # positions 1..k (keep 0 fixed)
        row = rows[r]
        lo = (row[pos - 1] + 1) if pos >= 1 else 1
        hi = (row[pos + 1] - 1) if pos < k else best_m + 30
        if hi <= lo:
            continue
        newval = random.randint(lo, hi)
        if newval == row[pos]:
            continue
        old = row[pos]
        row[pos] = newval
        c = _conflict_count(rows)
        m = max(max(rw) for rw in rows)
        if c < best_conf or (c == best_conf and m < best_m):
            best_conf = c
            best_m = m
        else:
            row[pos] = old
    return rows


def _sa_optimize(rows, k, t0):
    """Simulated annealing with fast integer conflict-counting. We optimize the
    binding (largest) row first: pick the row with the largest entry and propose a
    single-mark move there, accepting conflict-reducing moves always and
    conflict-increasing moves with a temperature-scaled probability to escape greedy
    local minima. Returns the best valid (or lowest-conflict) rows found."""
    rows = [list(r) for r in rows]
    best_conf = _conflict_count(rows)
    best_m = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    T = 8.0
    T_min = 0.01
    cool = 0.999
    it = 0
    while T > T_min:
        if _deadline_reached(t0, margin=3.0):
            break
        # binding row = the one with the largest entry
        bind = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[bind]
        pos = random.randrange(1, k + 1)
        lo = (row[pos - 1] + 1) if pos >= 1 else 1
        hi = (row[pos + 1] - 1) if pos < k else best_m + 40
        if hi <= lo:
            T *= cool
            continue
        newval = random.randint(lo, hi)
        if newval == row[pos]:
            T *= cool
            continue
        old = row[pos]
        row[pos] = newval
        c = _conflict_count(rows)
        m = max(max(rw) for rw in rows)
        dc = c - best_conf
        if dc <= 0:
            # accept (improvement or equal)
            if c < best_conf or (c == best_conf and m < best_m):
                best_conf = c
                best_m = m
                best_rows = [list(rw) for rw in rows]
        else:
            # accept with probability exp(-dc / T)
            if random.random() < math.exp(-dc / T):
                pass  # keep the move
            else:
                row[pos] = old
        T *= cool
        it += 1
    return best_rows


def construct(n_rows, k):
    t0 = _now()
    # 1) Greedy baseline (fast, valid).
    rows = _greedy_construct(n_rows, k)
    best = [list(r) for r in rows]
    best_m = max(max(r) for r in best)

    # 2) SA to escape greedy minima, optimizing the binding row first.
    if not _deadline_reached(t0):
        sa_rows = _sa_optimize(rows, k, t0)
        if sa_rows:
            c = _conflict_count(sa_rows)
            m = max(max(r) for r in sa_rows)
            if c == 0 and m < best_m:
                best = [list(r) for r in sa_rows]
                best_m = m

    # 3) Per-row local polish to refine (only accepts non-increasing moves).
    if not _deadline_reached(t0):
        polished = _local_polish(best, k, t0, iters=20000)
        c = _conflict_count(polished)
        m = max(max(r) for r in polished)
        if c == 0 and m < best_m:
            best = polished
            best_m = m

    # Safety: ensure a valid result is returned even if polish produced conflicts.
    c = _conflict_count(best)
    if c != 0:
        # fall back to the greedy baseline which is always valid
        best = _greedy_construct(n_rows, k)

    return best
