"""Difference triangle set (7,5): minimize largest entry with distinct differences."""
import random
import time

_DEADLINE = 150.0
_MARGIN = 12.0


def _row_diffs(row):
    return frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))


def _all_diffs(rows):
    out = set()
    for r in rows:
        out |= _row_diffs(r)
    return out


def _valid(rows, k):
    seen = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        d = _row_diffs(r)
        if len(d) != k * (k + 1) // 2:
            return False
        if seen & d:
            return False
        seen |= d
    return True


def _scope(rows):
    return max(r[-1] for r in rows)


def _greedy_row(used, k, rng, max_probe=40000):
    row = [0]
    x = 0
    tries = 0
    while len(row) < k + 1 and tries < max_probe:
        x += 1
        tries += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    if len(row) < k + 1:
        return None
    return row


def _random_greedy(n_rows, k, rng, max_probe=40000):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_probe)
        if row is None:
            return None
        rows.append(row)
    return rows


def _local_search(rows, k, rng, deadline):
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    n = len(rows)
    used = _all_diffs(rows)
    scope = best_scope

    def _try_move(ri, idx, new_val):
        nonlocal used, scope, best_scope, best_rows
        row = rows[ri]
        old = row[idx]
        if new_val == old:
            return False
        if idx == 0:
            return False
        if not (row[idx - 1] < new_val < row[idx + 1] if idx + 1 < len(row) else row[idx - 1] < new_val):
            return False
        # remove old diffs involving idx
        removed = set()
        for j in range(len(row)):
            if j != idx:
                removed.add(abs(row[j] - old))
        # compute new diffs involving idx
        added = set()
        for j in range(len(row)):
            if j != idx:
                added.add(abs(row[j] - new_val))
        # check added not in used minus removed
        conflict = (added - removed) & (used - removed)
        if conflict:
            return False
        # check internal: added must be unique among themselves (they are distinct since row strictly increasing and new_val fixed, differences to distinct j are distinct)
        used = (used - removed) | added
        row[idx] = new_val
        new_scope = max(r[-1] for r in rows)
        if new_scope < scope:
            scope = new_scope
            return True
        # else revert
        row[idx] = old
        used = (used - added) | removed
        return False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(20000):
            if time.time() > deadline:
                break
            ri = rng.randrange(n)
            idx = rng.randrange(1, k + 1)
            row = rows[ri]
            lo = row[idx - 1] + 1
            hi = (row[idx + 1] - 1) if idx + 1 < len(row) else scope
            if lo > hi:
                continue
            cand = rng.randint(lo, hi)
            if _try_move(ri, idx, cand):
                improved = True
                best_scope = scope
                best_rows = [list(r) for r in rows]

    return best_rows, best_scope


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    # Initial greedy
    best = _random_greedy(n_rows, k, rng)
    if best is None or not _valid(best, k):
        return best if best is not None else [[0]]
    best_scope = _scope(best)
    best_rows = [list(r) for r in best]

    # Local search on the initial solution
    best_rows, best_scope = _local_search(best_rows, k, rng, deadline)

    # Restart with randomized greedy, then local search, while time remains
    restarts = 0
    while time.time() < deadline:
        cand = _random_greedy(n_rows, k, rng, max_probe=20000)
        if cand is None or not _valid(cand, k):
            continue
        s = _scope(cand)
        if s < best_scope:
            best_rows = [list(r) for r in cand]
            best_scope = s
        # local search on this candidate
        lr, ls = _local_search(best_rows, k, rng, deadline)
        if ls < best_scope:
            best_scope = ls
            best_rows = lr
        restarts += 1

    return best_rows


if __name__ == "__main__":
    res = construct(7, 5)
    print(res)
    print(_scope(res), _valid(res, 5))
