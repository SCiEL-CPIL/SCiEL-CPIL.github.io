import time
import random

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under the 150 s kill


def _max_entry(rows):
    m = 0
    for r in rows:
        if r:
            v = r[-1]
            if v > m:
                m = v
    return m


def _all_diffs(rows):
    d = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            a = r[i]
            for j in range(i + 1, n):
                d.add(r[j] - a)
    return d


def _valid(rows):
    d = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            a = r[i]
            for j in range(i + 1, n):
                x = r[j] - a
                if x in d:
                    return False
                d.add(x)
    return True


def _greedy(n_rows, k, rng):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new = {x - a for a in row}
                if not (new & used):
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    rng = random.Random(12345)

    def stop():
        return time.time() > _DEADLINE

    # Strong baseline: pick the best of a few deterministic/randomized orders.
    best = None
    best_val = None
    for trial in range(6):
        if stop():
            break
        rows = _greedy(n_rows, k, rng)
        if not _valid(rows):
            continue
        v = _max_entry(rows)
        if best_val is None or v < best_val:
            best_val = v
            best = [list(r) for r in rows]
    if best is None:
        rows = _greedy(n_rows, k, rng)
        return rows

    # Hill-climbing: swap two entries inside a row; accept if still valid and scope shrinks.
    rows = [list(r) for r in best]
    cur = best_val
    rng2 = random.Random(777)

    def try_move():
        nonlocal cur
        r = rng2.randrange(n_rows)
        row = rows[r]
        i, j = rng2.sample(range(1, len(row)), 2)
        row[i], row[j] = row[j], row[i]
        if _valid(rows):
            v = _max_entry(rows)
            if v < cur:
                cur = v
                return True
        row[i], row[j] = row[j], row[i]
        return False

    # 1-opt: move a single entry to a smaller free position (reduces max without reordering).
    def try_move1():
        nonlocal cur
        r = rng2.randrange(n_rows)
        row = rows[r]
        idx = rng2.randrange(1, len(row))
        old = row[idx]
        lo = row[idx - 1] + 1
        hi = old - 1
        if hi < lo:
            return False
        for cand in (hi, lo, rng2.randrange(lo, hi + 1)):
            row[idx] = cand
            if _valid(rows):
                v = _max_entry(rows)
                if v < cur:
                    cur = v
                    return True
        row[idx] = old
        return False

    no_improve = 0
    while not stop():
        improved = try_move() or try_move1()
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 40000:
                # restart the random walk from current best to escape local minima
                no_improve = 0
        rows = rows  # keep reference

    return rows
