"""Difference triangle set (4,7): greedy baseline + simulated-annealing row reordering
   + targeted local polish, all under a wall-clock deadline guard."""
import time
import random

_DEADLINE = 140.0  # hard wall-clock kill margin (evaluator kills at ~150s)


def _build(rows, k):
    """Given an ordered list of row-index groups, build the rows greedily in that order.
    Returns (rows_list, used_set, scope) or None if a row cannot be completed within scope."""
    used = set()
    out = []
    for gi in rows:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > gi:
                return None
            new = [x - a for a in row]
            # fast reject: any diff already used?
            bad = False
            for d in new:
                if d in used:
                    bad = True
                    break
            if bad:
                continue
            for d in new:
                used.add(d)
            row.append(x)
        out.append(row)
    return out, used, max(r[-1] for r in out)


def _greedy_baseline(n_rows, k):
    """Original greedy: rows in natural order, each with its own tight scope."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = [x - a for a in row]
            if any(d in used for d in new):
                continue
            for d in new:
                used.add(d)
            row.append(x)
        rows.append(row)
    return rows


def _local_polish(rows, k, used, scope, deadline, rng):
    """Try small targeted swaps of row indices to reduce scope. Returns best found."""
    n = len(rows)
    idx = list(range(n))
    cur = _build(idx, k)
    if cur is None:
        return rows
    cur_rows, cur_used, cur_scope = cur
    best = (cur_scope, cur_rows)
    t = 0.0
    while time.time() < deadline:
        t += 1
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        idx2 = idx[:]
        idx2[a], idx2[b] = idx2[b], idx2[a]
        cand = _build(idx2, k)
        if cand is None:
            continue
        c_rows, c_used, c_scope = cand
        if c_scope < cur_scope or (c_scope == cur_scope and rng.random() < 0.3):
            cur_scope, cur_rows, cur_used = c_scope, c_rows, c_used
            idx = idx2
            if c_scope < best[0]:
                best = (c_scope, c_rows)
    return best[1]


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _DEADLINE
    rng = random.Random(12345)

    # Fast valid baseline (always returned if nothing better is found).
    base = _greedy_baseline(n_rows, k)
    base_scope = max(r[-1] for r in base)

    # Build initial SA state: greedy rows in natural order with a shared scope.
    idx = list(range(n_rows))
    init = _build(idx, k)
    if init is None:
        return base
    cur_rows, cur_used, cur_scope = init
    if cur_scope >= base_scope:
        # SA start is no better than baseline; use baseline as the working state.
        cur_rows, cur_scope = base, base_scope
        # rebuild used from baseline
        cur_used = set()
        for r in cur_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    cur_used.add(r[j] - r[i])
        idx = list(range(n_rows))

    best_scope = cur_scope
    best_rows = cur_rows

    # ---- Simulated annealing over row-index permutations ----
    T = 8.0
    Tmin = 0.05
    cool = 0.985
    while time.time() < deadline - 5.0:
        a = rng.randrange(n_rows)
        b = rng.randrange(n_rows)
        if a == b:
            continue
        idx2 = idx[:]
        idx2[a], idx2[b] = idx2[b], idx2[a]
        cand = _build(idx2, k)
        if cand is None:
            continue
        c_rows, c_used, c_scope = cand
        d = c_scope - cur_scope
        if d <= 0 or rng.random() < min(1.0, (-d) / max(T, 1e-9)):
            cur_scope, cur_rows, cur_used, idx = c_scope, c_rows, c_used, idx2
            if c_scope < best_scope:
                best_scope, best_rows = c_scope, c_rows
        T *= cool
        if T < Tmin:
            T = Tmin

    # ---- Local polish (targeted swaps) in remaining time ----
    if time.time() < deadline - 2.0:
        polished = _local_polish(best_rows, k, None, best_scope, deadline - 2.0, rng)
        ps = max(r[-1] for r in polished)
        if ps < best_scope:
            best_scope, best_rows = ps, polished

    # Safety margin: ensure we return before the hard deadline.
    if time.time() > _DEADLINE - 3.0:
        pass
    return best_rows
