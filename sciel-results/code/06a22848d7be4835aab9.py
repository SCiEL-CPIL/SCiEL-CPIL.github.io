"""Difference triangle set (7,5) - simulated annealing with worst-row re-optimization."""
import time
import random
import math


def _all_diffs(rows):
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                used.add(row[j] - ai)
    return used


def _max_entry(rows):
    m = 0
    for row in rows:
        if row:
            v = row[-1]
            if v > m:
                m = v
    return m


def _reopt_row(row, used, k):
    """Shrink a single row avoiding the fixed set `used` (diffs of other rows)."""
    n = k + 1
    best = list(row)
    best_max = best[-1]
    for _pass in range(6000):
        improved = False
        for idx in range(n - 1, 0, -1):
            lo = best[idx - 1] + 1
            hi = best[idx] - 1
            if hi < lo:
                continue
            cur = best[idx]
            for cand in range(lo, cur):
                ok = True
                for j in range(idx):
                    d = cand - best[j]
                    if d in used:
                        ok = False
                        break
                    for l in range(j + 1, idx):
                        if d == best[l] - best[j]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best[idx] = cand
                    improved = True
                    break
        if best[-1] < best_max:
            best_max = best[-1]
        if not improved:
            break
    if best[-1] < row[-1]:
        return best
    return None


def construct(n_rows, k):
    start = time.time()
    deadline = start + 150.0 - 8.0
    rng = random.Random(12345)

    def now():
        return time.time()

    # initial greedy rulers
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

    best_rows = [list(r) for r in rows]
    best_max = _max_entry(best_rows)

    cur_rows = [list(r) for r in rows]
    cur_max = _max_entry(cur_rows)
    cur_used = _all_diffs(cur_rows)

    T = float(max(1, best_max))
    T_end = 1e-3
    alpha = 0.9995

    step = 0
    reopt_every = 300
    last_reopt = 0

    def rebuild_used(skip):
        used = set()
        for ri, row in enumerate(cur_rows):
            if ri == skip:
                continue
            n = len(row)
            for i in range(n):
                ai = row[i]
                for j in range(i + 1, n):
                    used.add(row[j] - ai)
        return used

    while now() < deadline:
        step += 1

        if step - last_reopt >= reopt_every:
            last_reopt = step
            worst = max(range(n_rows), key=lambda i: cur_rows[i][-1])
            other_used = rebuild_used(worst)
            new_row = _reopt_row(cur_rows[worst], other_used, k)
            if new_row is not None:
                cur_rows[worst] = new_row
                cur_used = _all_diffs(cur_rows)
                new_max = _max_entry(cur_rows)
                cur_max = new_max
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(r) for r in cur_rows]

        ri = rng.randrange(n_rows)
        row = cur_rows[ri]
        n = len(row)
        idx = rng.randrange(1, n)
        span = max(3, int(0.05 * (row[-1] + 1)))
        delta = rng.randint(-span, span)
        cand = row[idx] + delta
        lo = row[idx - 1] + 1
        hi = row[idx + 1] - 1 if idx + 1 < n else 10 ** 9
        if cand < lo:
            cand = lo
        if cand > hi:
            cand = hi
        if cand == row[idx]:
            continue

        removed = set()
        added = set()
        for j in range(idx):
            removed.add(row[idx] - row[j])
            added.add(cand - row[j])
        for j in range(idx + 1, n):
            removed.add(row[j] - row[idx])
            added.add(row[j] - cand)

        # internal collision among fixed entries
        ok = True
        for j in range(idx):
            for l in range(j + 1, n):
                if j == idx or l == idx:
                    continue
                # diff among fixed entries
                pass
        # check added against fixed-internal diffs
        for a in range(n):
            if a == idx:
                continue
            for b in range(a + 1, n):
                if b == idx:
                    continue
                fd = row[b] - row[a]
                if fd in added:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        global_block = cur_used - removed
        if added & global_block:
            continue

        # apply
        row[idx] = cand
        cur_used = _all_diffs(cur_rows)
        new_max = _max_entry(cur_rows)

        d_obj = new_max - cur_max
        if d_obj <= 0:
            cur_max = new_max
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in cur_rows]
        else:
            if T > T_end and rng.random() < math.exp(-d_obj / max(T, 1e-9)):
                cur_max = new_max
        T *= alpha

    return best_rows
