"""Difference triangle set (4,7): fresh multi-start packing design."""
import time

_DEADLINE = 115.0  # wall-clock safety margin (evaluator kills at 150s)
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _diffs(row):
    d = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ri)
    return d


def _valid(rows):
    used = set()
    for row in rows:
        d = _diffs(row)
        if d & used:
            return False
        used |= d
    return True


def _greedy(n_rows, k, seed=0):
    """Deterministic smallest-next-mark (the seed behavior); always valid & fast."""
    import random
    rng = random.Random(seed)
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _build_row(used, k, max_val, rng, time_budget):
    """Backtracking + lookahead: build a row of k+1 increasing ints in [0, max_val]
    starting at 0, whose pairwise differences avoid `used`."""
    target = k + 1
    deadline = time.time() + time_budget

    def rec(row):
        if time.time() > deadline:
            return None
        if len(row) == target:
            return list(row)
        last = row[-1]
        # candidate next marks: last+1 .. max_val
        for x in range(last + 1, max_val + 1):
            new = {x - a for a in row}
            if new & used:
                continue
            row.append(x)
            res = rec(row)
            if res is not None:
                return res
            row.pop()
        return None

    return rec([0])


def _search(n_rows, k, max_val, seed, time_budget):
    rng = random.Random(seed)
    used, rows = set(), []
    # order rows by a random permutation for diversity
    for _ in range(n_rows):
        row = _build_row(used, k, max_val, rng, time_budget)
        if row is None:
            return None
        d = _diffs(row)
        used |= d
        rows.append(row)
    return rows


def _lower_bound(n_rows, k):
    total = n_rows * (k * (k + 1) // 2)
    m = 0
    while m * (m + 1) // 2 < total:
        m += 1
    return m


def construct(n_rows, k):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 115.0

    # fast guaranteed fallback
    fallback = _greedy(n_rows, k, seed=0)
    best = fallback
    best_val = max(max(r) for r in fallback)

    lb = _lower_bound(n_rows, k)
    if lb < 2:
        lb = 2

    # attempt increasing largest-entry bounds
    m = lb
    seed = 0
    while m <= best_val and _time_left() > 3.0:
        # a couple of restarts per bound; keep total restart time bounded
        restarts = 3
        found = None
        for _ in range(restarts):
            if _time_left() < 3.0:
                break
            per = max(0.05, min(2.0, _time_left() / (restarts + 1)))
            res = _search(n_rows, k, m, seed, per)
            seed += 1
            if res is not None:
                found = res
                break
        if found is not None:
            best = found
            best_val = m
            break
        m += 1

    return best
