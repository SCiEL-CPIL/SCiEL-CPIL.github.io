"""Difference triangle set (n_rows, k): local-search design."""
import time
import random

_DEADLINE = 140.0  # safety margin under the 150s wall-clock kill


def _row_gaps(row):
    """row (list of k+1 ints starting at 0) -> list of k positive gaps."""
    return [row[i + 1] - row[i] for i in range(len(row) - 1)]


def _row_from_gaps(gaps):
    """gaps (list of k positive ints) -> row of k+1 ints starting at 0."""
    row = [0]
    acc = 0
    for g in gaps:
        acc += g
        row.append(acc)
    return row


def _all_diffs(row):
    """All positive differences a[j]-a[i] (i<j) of a row."""
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d


def _diffs_from_gaps(gaps):
    """All positive differences of the row built from gaps, via prefix sums."""
    prefix = [0]
    acc = 0
    for g in gaps:
        acc += g
        prefix.append(acc)
    d = []
    n = len(prefix)
    for i in range(n):
        ai = prefix[i]
        for j in range(i + 1, n):
            d.append(prefix[j] - ai)
    return d


def _random_gaps(k, max_span):
    """Randomly partition max_span into k positive parts."""
    if k <= 0:
        return []
    cuts = sorted(random.sample(range(1, max_span), k - 1))
    prev = 0
    gaps = []
    for c in cuts:
        gaps.append(c - prev)
        prev = c
    gaps.append(max_span - prev)
    return gaps


def construct(n_rows, k):
    start = time.time()

    # Upper bound for the search: greedy ruler scope gives a valid starting point
    # and a ceiling; we never need to exceed it.
    greedy_used = set()
    greedy_rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & greedy_used:
                row.append(x)
                greedy_used |= new
        greedy_rows.append(row)
    greedy_scope = max(max(r) for r in greedy_rows)

    # Number of distinct positive differences in one row
    n_diff = k * (k + 1) // 2

    def build_state(gaps_list, scope):
        """Return (rows, diff_multiset_dict, scope) for a set of gap-lists."""
        rows = [_row_from_gaps(g) for g in gaps_list]
        diff_count = {}
        for g in gaps_list:
            for d in _diffs_from_gaps(g):
                diff_count[d] = diff_count.get(d, 0) + 1
        return rows, diff_count, scope

    best_rows = greedy_rows
    best_scope = greedy_scope

    def _is_valid(diff_count):
        return all(c == 1 for c in diff_count.values())

    # Seed initial state from greedy (already valid)
    init_gaps = [_row_gaps(r) for r in greedy_rows]
    rows, diff_count, scope = build_state(init_gaps, greedy_scope)
    assert _is_valid(diff_count)

    T0 = greedy_scope  # initial temperature
    cool = 0.995
    T = T0
    max_gap = greedy_scope + 10

    def _move_swap(gaps_list, diff_count, row_i, a, b):
        """Swap gaps a and b in row row_i. Returns (ok, new_diff_count, new_scope)."""
        g = gaps_list[row_i]
        old = _diffs_from_gaps(g)
        # undo old
        for d in old:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        # apply swap
        g[a], g[b] = g[b], g[a]
        new = _diffs_from_gaps(g)
        for d in new:
            diff_count[d] = diff_count.get(d, 0) + 1
        new_scope = max(sum(g2) for g2 in gaps_list)
        return True, diff_count, new_scope

    def _move_replace(gaps_list, diff_count, row_i, pos, new_g):
        """Replace gap at pos in row row_i with new_g. Returns (ok, new_diff_count, new_scope)."""
        g = gaps_list[row_i]
        old = _diffs_from_gaps(g)
        for d in old:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        g[pos] = new_g
        new = _diffs_from_gaps(g)
        for d in new:
            diff_count[d] = diff_count.get(d, 0) + 1
        new_scope = max(sum(g2) for g2 in gaps_list)
        return True, diff_count, new_scope

    # Annealing loop
    stall = 0
    restarts = 0
    while time.time() - start < _DEADLINE:
        row_i = random.randrange(n_rows)
        if random.random() < 0.6 and k >= 2:
            a = random.randrange(k)
            b = random.randrange(k)
            if a == b:
                continue
            # snapshot to allow rollback
            old_diff = dict(diff_count)
            old_gaps_row = list(gaps_list[row_i])
            _, diff_count, new_scope = _move_swap(gaps_list, diff_count, row_i, a, b)
            valid = _is_valid(diff_count)
            if valid:
                if new_scope < scope:
                    scope = new_scope
                    stall = 0
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [_row_from_gaps(g) for g in gaps_list]
                else:
                    stall += 1
                    if stall > 2000:
                        # restart from best
                        gaps_list = [_row_gaps(r) for r in best_rows]
                        rows, diff_count, scope = build_state(gaps_list, best_scope)
                        T = T0
                        stall = 0
                        restarts += 1
            else:
                # rollback
                gaps_list[row_i] = old_gaps_row
                diff_count = old_diff
                stall += 1
                if stall > 2000:
                    gaps_list = [_row_gaps(r) for r in best_rows]
                    rows, diff_count, scope = build_state(gaps_list, best_scope)
                    T = T0
                    stall = 0
                    restarts += 1
        else:
            pos = random.randrange(k)
            new_g = random.randint(1, max_gap)
            old_diff = dict(diff_count)
            old_gaps_row = list(gaps_list[row_i])
            _, diff_count, new_scope = _move_replace(gaps_list, diff_count, row_i, pos, new_g)
            valid = _is_valid(diff_count)
            if valid:
                if new_scope < scope:
                    scope = new_scope
                    stall = 0
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [_row_from_gaps(g) for g in gaps_list]
                else:
                    # accept worse/equal with probability
                    if random.random() < 0.02:
                        stall += 1
                    else:
                        gaps_list[row_i] = old_gaps_row
                        diff_count = old_diff
                        stall += 1
                if stall > 2000:
                    gaps_list = [_row_gaps(r) for r in best_rows]
                    rows, diff_count, scope = build_state(gaps_list, best_scope)
                    T = T0
                    stall = 0
                    restarts += 1
            else:
                gaps_list[row_i] = old_gaps_row
                diff_count = old_diff
                stall += 1
                if stall > 2000:
                    gaps_list = [_row_gaps(r) for r in best_rows]
                    rows, diff_count, scope = build_state(gaps_list, best_scope)
                    T = T0
                    stall = 0
                    restarts += 1
        T *= cool

    # Final validation of best_rows
    final_used = set()
    for r in best_rows:
        for d in _all_diffs(r):
            if d in final_used:
                # should not happen; fall back to greedy
                return greedy_rows
            final_used.add(d)
    return best_rows
