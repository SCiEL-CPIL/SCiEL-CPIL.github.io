"""Difference triangle set (4,7): minimize the largest entry.

Structural representation: each row is a frozenset of its positive
differences. Conflict counting = set intersection between row diff-sets,
which is much cheaper than rebuilding a global difference Counter per
mutating element.  We keep the fast conflict machinery + binding-row-first
SA + local polish of the seed, but run it on top of the diff-set
representation and a multi-start greedy baseline.
"""
import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin under 150s wall-clock


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _diffs_of_row(row):
    """Return a frozenset of all positive differences in one row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _validate(rows):
    """True iff all positive differences are distinct across all rows."""
    seen = set()
    for row in rows:
        d = _diffs_of_row(row)
        if len(d) != len(row) * (len(row) - 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True


def _greedy_build(n_rows, k, rng, order=None):
    """Build rows greedily in the given order (smallest available x first)."""
    if order is None:
        order = list(range(n_rows))
    used = set()
    rows = [None] * n_rows
    for r in order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows


def _multi_start_greedy(n_rows, k, rng, budget):
    """Try several greedy orders, keep the one with the smallest max entry."""
    best = None
    best_max = None
    while _time_left() > budget:
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, rng, order)
        m = max(max(r) for r in rows)
        if best is None or m < best_max:
            best = rows
            best_max = m
    return best, best_max


def _sa_optimize(rows, n_rows, k, rng, time_budget):
    """
    Simulated annealing over the diff-set representation.
    State: list of rows + list of per-row diff frozensets.
    Move: change one element (>=1) of one row by a small delta.
    Energy: number of cross-row conflicts (sum of |diffset_i & diffset_j|).
    We accept conflict-reducing moves and, when conflict-free, scope-reducing
    moves; we allow small conflict excursions at high T to escape local minima.
    """
    rows = [list(r) for r in rows]
    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]

    def conflict_total(rds):
        tot = 0
        for i in range(n_rows):
            di = rds[i]
            for j in range(i + 1, n_rows):
                tot += len(di & rds[j])
        return tot

    def total_max(rds_rows):
        m = 0
        for r in rds_rows:
            if r[-1] > m:
                m = r[-1]
        return m

    cur_conf = conflict_total(row_diffs)
    best_conf = cur_conf
    best_rows = [list(r) for r in rows]
    best_max = total_max(rows)

    start = time.time()
    T = max(2.0, best_max * 0.15)
    T_min = 0.3
    cooling = 0.9985

    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        # Bias toward the row with the largest entry (binding row).
        r = rng.randrange(n_rows)
        if rng.random() < 0.6:
            # pick the binding row explicitly
            bi = 0
            for x in range(1, n_rows):
                if rows[x][-1] > rows[bi][-1]:
                    bi = x
            r = bi
        row = rows[r]
        if len(row) <= 1:
            T *= cooling
            continue
        pos = rng.randrange(1, len(row))
        old_val = row[pos]

        span = max(1, min(20, int(T)))
        delta = rng.randint(-span, span)
        new_val = old_val + delta
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        if new_val == old_val:
            continue

        row_copy = list(row)
        row_copy[pos] = new_val
        new_rd = _diffs_of_row(row_copy)
        if len(new_rd) != k * (k + 1) // 2:
            T *= cooling
            continue

        # Conflict delta: only this row's diffset changes.
        old_conf_with_others = 0
        new_conf_with_others = 0
        for other_r in range(n_rows):
            if other_r == r:
                continue
            od = row_diffs[other_r]
            old_conf_with_others += len(row_diffs[r] & od)
            new_conf_with_others += len(new_rd & od)
        delta_conf = new_conf_with_others - old_conf_with_others
        new_conf = cur_conf + delta_conf

        new_row_max = max(new_rd)  # row[-1] new value; max diff == last element
        # total scope
        other_max = 0
        for x in range(n_rows):
            if x != r:
                if rows[x][-1] > other_max:
                    other_max = rows[x][-1]
        cur_scope = max(old_val if old_val > row[-1] else row[-1], other_max)
        new_scope = max(new_val if new_val > row[-1] else row[-1], other_max)

        if delta_conf < 0:
            accept = True
        elif delta_conf == 0:
            if new_scope <= cur_scope:
                accept = True
            else:
                accept = rng.random() < 0.25
        else:
            if delta_conf <= 3 and rng.random() < 0.02 * (T / 100.0):
                accept = True
            else:
                accept = False

        if accept:
            row[pos] = new_val
            row_diffs[r] = new_rd
            cur_conf = new_conf
            if cur_conf < best_conf or (cur_conf == best_conf and new_scope < best_max):
                best_conf = cur_conf
                best_max = new_scope
                best_rows = [list(x) for x in rows]
        T *= cooling

    return best_rows, best_max


def _local_search_reduce(rows, n_rows, k, rng, time_budget):
    """Greedy reduction: try to lower the largest elements, keeping validity."""
    rows = [list(r) for r in rows]
    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]
    best_rows = [list(r) for r in rows]
    best_max = total_max(rows)

    def total_max(rs):
        m = 0
        for r in rs:
            if r[-1] > m:
                m = r[-1]
        return m

    def conflict_total(rds):
        tot = 0
        for i in range(n_rows):
            di = rds[i]
            for j in range(i + 1, n_rows):
                tot += len(di & rds[j])
        return tot

    start = time.time()
    while time.time() - start < time_budget and _time_left() > 3.0:
        # target the binding row
        bi = 0
        for x in range(1, n_rows):
            if rows[x][-1] > rows[bi][-1]:
                bi = x
        row = rows[bi]
        improved = False

        # try to reduce each element (from the end) by 1.., keeping increasing & valid
        for pos in range(len(row) - 1, 0, -1):
            old = row[pos]
            for new_val in range(old - 1, row[pos - 1], -1):
                if pos < k and new_val >= row[pos + 1]:
                    continue
                row_copy = list(row)
                row_copy[pos] = new_val
                new_rd = _diffs_of_row(row_copy)
                if len(new_rd) != k * (k + 1) // 2:
                    continue
                # check conflicts with other rows
                conflict = False
                for other_r in range(n_rows):
                    if other_r == bi:
                        continue
                    if new_rd & row_diffs[other_r]:
                        conflict = True
                        break
                if not conflict:
                    row[pos] = new_val
                    row_diffs[bi] = new_rd
                    nm = total_max(rows)
                    if nm < best_max:
                        best_max = nm
                        best_rows = [list(x) for x in rows]
                    improved = True
                    break
        if not improved:
            # random small perturb then re-reduce to escape
            r = rng.randrange(n_rows)
            if len(rows[r]) <= 2:
                break
            pos = rng.randrange(1, len(rows[r]) - 1)
            row = rows[r]
            old = row[pos]
            for new_val in range(old + 1, old + 6):
                if pos < k and new_val >= row[pos + 1]:
                    continue
                row_copy = list(row)
                row_copy[pos] = new_val
                new_rd = _diffs_of_row(row_copy)
                if len(new_rd) != k * (k + 1) // 2:
                    continue
                conflict = False
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    if new_rd & row_diffs[other_r]:
                        conflict = True
                        break
                if not conflict:
                    row[pos] = new_val
                    row_diffs[r] = new_rd
                    break

    return best_rows, best_max


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct."""
    rng = random.Random(12345)

    # Multi-start greedy baseline (cheap, a few seconds).
    rows, cur_max = _multi_start_greedy(n_rows, k, rng, budget=6.0)
    best_rows = [list(r) for r in rows]
    best_max = cur_max

    # SA phase: most of the budget.
    rows, m = _sa_optimize(rows, n_rows, k, rng, time_budget=110.0)
    if m < best_max:
        best_max = m
        best_rows = [list(r) for r in rows]

    # Local search reduction phase.
    rows, m = _local_search_reduce(best_rows, n_rows, k, rng, time_budget=20.0)
    if m < best_max:
        best_max = m
        best_rows = [list(r) for r in rows]

    # Final safety: ensure validity; fall back to a fresh greedy if broken.
    if not _validate(best_rows):
        best_rows = _greedy_build(n_rows, k, rng)

    return best_rows
