"""Difference triangle set (n_rows, k).

Fresh design: modelled as a constraint-satisfaction / local-search problem
rather than the seed's one-pass greedy. We minimise the largest entry (scope)
while keeping every positive intra-row difference globally distinct.
"""
import time
import random
import numpy as np

_DEADLINE = 150.0          # hard wall-clock budget
_TIME_MARGIN = 1.0         # safety margin (seconds)
_RNG = random.Random(20250613)


def _deltas(row):
    """All positive intra-row differences of a strictly-increasing row."""
    out = []
    for i in range(len(row)):
        ri = row[i]
        for j in range(i + 1, len(row)):
            out.append(row[j] - ri)
    return out


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows):
    seen = set()
    for r in rows:
        if r[0] != 0:
            return False
        for i in range(len(r)):
            if i and r[i] <= r[i - 1]:
                return False
            ri = r[i]
            for j in range(i + 1, len(r)):
                d = r[j] - ri
                if d in seen:
                    return False
                seen.add(d)
    return True


def _greedy_row(used, k):
    """Smallest strictly-increasing row from 0 whose new differences are unused."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _init_greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _init_random(n_rows, k, scope):
    """Random feasible-in-shape rows: strictly increasing, start 0, within scope."""
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            lo = row[-1] + 1
            hi = scope
            row.append(_RNG.randint(lo, hi))
        rows.append(row)
    return rows


def _local_search(rows, n_rows, k, scope, rng, deadline):
    """Randomised local search minimising the largest entry.

    Moves: pick a random row and one of its non-zero entries, try a new value
    that keeps the row strictly increasing and within the current scope, and
    accept iff no global difference collision is introduced.
    Returns the best valid row set found (or None if none was valid).
    """
    n = k + 1
    cur = [list(r) for r in rows]
    cur_scope = max(max(r) for r in cur)

    # Precompute difference positions (i, j) for each row, i < j.
    pos = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best = None
    best_scope = None
    stuck = 0

    while time.time() < deadline:
        r = rng.randrange(n_rows)
        c = rng.randrange(1, n)          # never touch the leading 0
        old = cur[r][c]

        # candidate new values within [lo, hi], strictly increasing, within scope
        lo = cur[r][c - 1] + 1
        hi = min(scope, cur[r][c + 1] - 1) if c < n - 1 else scope
        if hi < lo:
            continue
        cands = list(range(lo, hi + 1))
        rng.shuffle(cands)

        # current deltas of row r (list parallel to pos)
        rd = cur[r]
        base = [rd[j] - rd[i] for (i, j) in pos]

        # global set of all differences currently in use
        used = set()
        for rr in range(n_rows):
            row = cur[rr]
            for (i, j) in pos:
                used.add(row[j] - row[i])

        moved = False
        for nv in cands:
            if nv == old:
                continue
            # affected deltas: those where c is i or j
            new_set = set()
            ok = True
            for idx, (i, j) in enumerate(pos):
                if i == c:
                    d = rd[j] - nv
                elif j == c:
                    d = nv - rd[i]
                else:
                    d = base[idx]
                if d in new_set:
                    ok = False
                    break
                new_set.add(d)
            if not ok:
                continue
            # collision check against the rest of the rows
            rest = used - new_set
            if new_set & rest:
                continue
            # accept
            cur[r][c] = nv
            moved = True
            break

        if moved:
            stuck = 0
            sc = max(max(row) for row in cur)
            if _valid(cur):
                if best_scope is None or sc < best_scope:
                    best_scope = sc
                    best = [list(row) for row in cur]
        else:
            stuck += 1
            if stuck > 400:
                break

    return best


def construct(n_rows, k):
    deadline = time.time() + _DEADLINE - _TIME_MARGIN

    # deterministic baseline (always valid)
    base_rows = _init_greedy(n_rows, k)
    best_rows = base_rows
    best_scope = _scope(base_rows)

    rng = random.Random(_RNG.randrange(1 << 30))

    while time.time() < deadline:
        scope = _RNG.randint(k + 1, best_scope)
        rows = _init_random(n_rows, k, scope)
        res = _local_search(rows, n_rows, k, scope, rng, deadline)
        if res is not None:
            sc = _scope(res)
            if sc < best_scope:
                best_scope = sc
                best_rows = res

    return best_rows
