"""Difference triangle set (7,5): minimize the largest entry."""
import random
import time

# EVOLVE-BLOCK-START

def _used_of(row):
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _valid(rows, k):
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for a, b in zip(r, r[1:]):
            if a >= b:
                return False
        d = _used_of(r)
        if d & used:
            return False
        used |= d
    return True

def _make_rows(n_rows, k, rng, max_start=400):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            if x > max_start:
                return None
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _reduce_element(row, j, used, k):
    """Minimize row[j] (keeping order and uniqueness). Returns (row, used, changed)."""
    lo = (row[j - 1] + 1) if j > 0 else 1
    hi = row[j] - 1
    old = row[j]
    for x in range(lo, hi + 1):
        if x == old:
            continue
        new = {x - a for a in row[:j]} | {row[t] - x for t in range(j + 1, len(row))}
        if not (new & used):
            row[j] = x
            used.discard(old - row[j - 1]) if j > 0 else None
            return row, used, True
    return row, used, False

def _reduce_row(row, used, k):
    """Minimize each element of a single row (recompute used each change)."""
    changed = True
    while changed:
        changed = False
        for j in range(len(row) - 1, 0, -1):
            row, used, c = _reduce_element(row, j, used, k)
            if c:
                used = _used_of(row)
                changed = True
    return row, used

def _reduce_all(rows, k):
    used = set()
    for r in rows:
        used |= _used_of(r)
    improved = True
    while improved:
        improved = False
        for idx, r in enumerate(rows):
            used = set()
            for t, rt in enumerate(rows):
                if t != idx:
                    used |= _used_of(rt)
            newr, used, changed = _reduce_row(r, used, k)
            if changed:
                rows[idx] = newr
                improved = True
    return rows

def _swap_rows(rows, i, j, k):
    used = set()
    for t, r in enumerate(rows):
        if t not in (i, j):
            used |= _used_of(r)
    ri, rj = rows[i], rows[j]
    used2 = set()
    for t, r in enumerate(rows):
        if t not in (i, j):
            used2 |= _used_of(r)
    # try swapping the two rows' last (largest) entries if it reduces max
    if len(ri) == len(rj) and ri[-1] != rj[-1]:
        cand = [list(r) for r in rows]
        cand[i][-1], cand[j][-1] = rj[-1], ri[-1]
        if _valid(cand, k):
            rows[i][-1], rows[j][-1] = rj[-1], ri[-1]
            return True
    return False

def _rebuild_row(rows, idx, k, rng):
    """Rebuild one row from scratch using the used diffs of the others."""
    used = set()
    for t, r in enumerate(rows):
        if t != idx:
            used |= _used_of(r)
    row, x = [0], 0
    while len(row) < k + 1:
        x += 1
        if x > 400:
            return False
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    rows[idx] = row
    return True

def _anneal(rows, k, rng, deadline):
    used = set()
    for r in rows:
        used |= _used_of(r)
    cur = max(max(r) for r in rows)
    best = [list(r) for r in rows]
    best_val = cur
    T = 20.0
    step = 0
    while time.time() < deadline:
        step += 1
        if step % 200 == 0:
            T *= 0.97
        idx = rng.randrange(len(rows))
        # choose an element to perturb
        j = rng.randrange(1, len(rows[idx]))
        cand = [list(r) for r in rows]
        old = cand[idx][j]
        delta = rng.choice([-1, 1]) * rng.randint(1, 4)
        cand[idx][j] = old + delta
        ok = True
        for t in range(len(cand[idx])):
            if cand[idx][t] < 0 or cand[idx][t] >= cand[idx][t + 1]:
                ok = False
                break
        if ok:
            used2 = set()
            for t, r in enumerate(cand):
                if t != idx:
                    used2 |= _used_of(r)
            d = _used_of(cand[idx])
            if d & used2:
                ok = False
        if ok:
            newval = max(max(r) for r in cand)
            if newval <= cur or rng.random() < min(1.0, T / max(1, cur)):
                rows = cand
                cur = newval
                if cur < best_val:
                    best_val = cur
                    best = [list(r) for r in rows]
        else:
            # occasionally rebuild the row to escape
            if rng.random() < 0.05:
                _rebuild_row(rows, idx, k, rng)
                used = set()
                for r in rows:
                    used |= _used_of(r)
                cur = max(max(r) for r in rows)
                if cur < best_val:
                    best_val = cur
                    best = [list(r) for r in rows]
    return best, best_val

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    rng = random.Random(12345)

    best = None
    best_val = float('inf')

    # Build a starting population via randomized greedy
    for attempt in range(30):
        if time.time() > deadline:
            break
        rows = _make_rows(n_rows, k, rng, max_start=300)
        if rows is None:
            continue
        if not _valid(rows, k):
            continue
        rows = _reduce_all(rows, k)
        val = max(max(r) for r in rows)
        if val < best_val:
            best_val = val
            best = [list(r) for r in rows]

    if best is None:
        # fall back to a deterministic greedy
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

    # Local search: element reduction + row swaps + simulated annealing
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        rows = [list(r) for r in best]
        rows = _reduce_all(rows, k)
        val = max(max(r) for r in rows)
        if val < best_val:
            best_val = val
            best = [list(r) for r in rows]
            improved = True
        # row swaps
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if _swap_rows(rows, i, j, k):
                    rows = _reduce_all(rows, k)
                    val = max(max(r) for r in rows)
                    if val < best_val:
                        best_val = val
                        best = [list(r) for r in rows]
                        improved = True

    # Final polish with annealing if time remains
    if time.time() < deadline - 10.0:
        rows = [list(r) for r in best]
        best, best_val = _anneal(rows, k, rng, deadline - 5.0)

    # Final reduction pass
    rows = _reduce_all([list(r) for r in best], k)
    val = max(max(r) for r in rows)
    if val < best_val:
        best_val = val
        best = rows

    return best

# EVOLVE-BLOCK-END
