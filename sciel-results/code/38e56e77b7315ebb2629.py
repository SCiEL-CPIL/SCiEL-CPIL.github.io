"""Difference triangle set (7,5) - greedy + re-insertion local search."""
import time
import random
import sys

DEADLINE = 140.0  # safety margin under the 150s kill


def _diffs(row):
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _valid(row, used):
    return not (_diffs(row) & used)


def _maxval(rows):
    return max(max(r) for r in rows)


def _greedy(n_rows, k, rng=None):
    """Plain greedy: smallest next mark whose differences are unused."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, rng, bias=0.35, gap=3):
    """Greedy that occasionally jumps forward to escape local minima."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
                if rng.random() < bias:
                    x += rng.randint(1, gap)
        rows.append(row)
    return rows


def _reinsert_local(rows, k, rng, iters=4000):
    """Re-insertion: drop a mark, reinsert at the smallest legal spot (or a random
    later spot), accepting only if the new max is no worse. Escapes local minima
    while never worsening the objective."""
    best = _maxval(rows)
    for _ in range(iters):
        ri = rng.randrange(len(rows))
        row = rows[ri]
        if len(row) <= 1:
            continue
        pos = rng.randrange(1, len(row))
        removed = row.pop(pos)
        used = set()
        for r in rows:
            used |= _diffs(r)
        row.append(removed)  # restore by default
        cand = removed
        ok = False
        # try the smallest legal insertion position at/after `pos`
        for t in range(pos, len(row) + 1):
            v = t
            while True:
                trial = row[:t] + [v] + row[t:]
                if _valid(trial, used):
                    cand = v
                    ok = True
                    break
                v += 1
                if v > best + 2 * k + 10:
                    break
            if ok:
                break
        if not ok:
            continue
        # sometimes try a random later spot for more exploration
        if rng.random() < 0.2:
            t = rng.randrange(pos, len(row) + 1)
            v = rng.randint(cand, best + 2 * k + 10)
            trial = row[:t] + [v] + row[t:]
            if _valid(trial, used) and max(trial) <= best:
                cand = v
                ok = True
        if not ok:
            continue
        # apply
        row.pop()  # drop restored `removed`
        newrow = row[:pos] + [cand] + row[pos:]
        if newrow != row:
            if max(newrow) <= best:
                rows[ri] = newrow
                if max(newrow) < best:
                    best = max(newrow)
    return rows


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(12345)

    # Baseline plain greedy (deterministic) as the floor.
    best = _greedy(n_rows, k)
    best_val = _maxval(best)

    # Biased randomized constructions to tighten the initial scope.
    while time.time() < deadline - 20:
        cand = _randomized_greedy(n_rows, k, rng)
        v = _maxval(cand)
        if v < best_val:
            best, best_val = cand, v
            # quick polish
            if time.time() < deadline - 10:
                cand2 = _reinsert_local([r[:] for r in best], k, rng, iters=1500)
                v2 = _maxval(cand2)
                if v2 <= best_val:
                    best, best_val = cand2, v2
        if time.time() > deadline - 20:
            break

    # Final high-precision polish pass with more iterations.
    if time.time() < deadline - 3:
        polished = _reinsert_local([r[:] for r in best], k, rng, iters=8000)
        v = _maxval(polished)
        if v <= best_val:
            best = polished
            best_val = v

    # Guarantee a valid, strictly-increasing, 0-starting structure.
    for r in best:
        r.sort()
    return best
