"""Difference triangle set (7,5): minimize the largest entry over 7 rows of 6 marks."""
import time
import random
import sys

_DEADLINE = 140.0  # safety margin under 150s wall-clock kill

def _all_row_diffs(row):
    d = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ri)
    return d

def _is_valid(rows):
    used = set()
    for row in rows:
        if row[0] != 0:
            return False
        for a, b in zip(row, row[1:]):
            if b <= a:
                return False
        ds = _all_row_diffs(row)
        if ds & used:
            return False
        used |= ds
    return True

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy_row(k, used, start=1):
    """Build one row of k+1 marks starting at 0; all its diffs disjoint from `used`."""
    row = [0]
    x = start
    while len(row) < k + 1:
        # find smallest x such that diffs to existing row are fresh
        ok = True
        for a in row:
            if (x - a) in used:
                ok = False
                break
        if ok:
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
            x += 1
        else:
            x += 1
    return row

def _greedy_rows(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(k, used))
    return rows

def _random_golomb_row(k, max_mark, rng):
    """Random row of k+1 increasing ints in [0, max_mark], all internal diffs distinct."""
    while True:
        row = [0] + sorted(rng.sample(range(1, max_mark + 1), k))
        if len(set(_all_row_diffs(row))) == len(_all_row_diffs(row)):
            return row

def _try_insert(rows, used, idx, row, new_x):
    """Return (new_used, ok) if replacing row[idx] with new_x keeps validity."""
    old = rows[idx]
    if new_x <= old[idx - 1] or new_x >= old[idx + 1]:
        return None, False
    new_row = old[:idx] + [new_x] + old[idx + 1:]
    new_ds = _all_row_diffs(new_row)
    # remove old diffs, add new, check disjoint with others
    others = used - _all_row_diffs(old)
    if new_ds & others:
        return None, False
    return used - _all_row_diffs(old) | new_ds, True

def _local_search(rows, n_rows, k, rng, deadline):
    """Move/swap marks to reduce scope. Returns best rows found."""
    used = set()
    for r in rows:
        used |= _all_row_diffs(r)
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    t0 = time.time()

    # Precompute per-row diff sets for speed
    row_diffs = [_all_row_diffs(r) for r in rows]

    def check_move(ri, pos, new_x):
        row = rows[ri]
        if new_x <= row[pos - 1] or new_x >= row[pos + 1]:
            return False
        new_row = row[:pos] + [new_x] + row[pos + 1:]
        new_ds = _all_row_diffs(new_row)
        others = used - row_diffs[ri]
        return not (new_ds & others)

    while time.time() < deadline:
        progress = False
        for ri in range(n_rows):
            if time.time() >= deadline:
                break
            row = rows[ri]
            for pos in range(1, len(row) - 1):
                if time.time() >= deadline:
                    break
                # try moving this mark to a nearby smaller value to reduce scope
                current = row[pos]
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1
                if lo >= hi:
                    continue
                # try a few candidate positions, biased toward lower to reduce scope
                candidates = []
                if current > lo:
                    candidates.append(lo)
                    candidates.append((lo + current) // 2)
                for _ in range(3):
                    c = rng.randint(lo, hi)
                    candidates.append(c)
                for c in candidates:
                    if c == current:
                        continue
                    if check_move(ri, pos, c):
                        # commit
                        old_ds = row_diffs[ri]
                        new_row = row[:pos] + [c] + row[pos + 1:]
                        new_ds = _all_row_diffs(new_row)
                        used = used - old_ds | new_ds
                        row_diffs[ri] = new_ds
                        rows[ri] = new_row
                        s = _scope(rows)
                        if s < best_scope:
                            best = [list(r) for r in rows]
                            best_scope = s
                        progress = True
                        break
        if not progress:
            # try a random perturbation to escape local optima
            ri = rng.randrange(n_rows)
            pos = rng.randrange(1, len(rows[ri]) - 1)
            row = rows[ri]
            lo, hi = row[pos - 1] + 1, row[pos + 1] - 1
            if hi - lo >= 2:
                c = rng.randint(lo, hi)
                if c != row[pos] and check_move(ri, pos, c):
                    old_ds = row_diffs[ri]
                    new_row = row[:pos] + [c] + row[pos + 1:]
                    new_ds = _all_row_diffs(new_row)
                    used = used - old_ds | new_ds
                    row_diffs[ri] = new_ds
                    rows[ri] = new_row
                    s = _scope(rows)
                    if s < best_scope:
                        best = [list(r) for r in rows]
                        best_scope = s
                    progress = True
        if not progress:
            break
    return best

def _improve_with_restarts(n_rows, k, rng, deadline):
    """Multiple random restarts with local search; keep best scope."""
    t0 = time.time()
    overall_best = None
    overall_scope = float('inf')

    # Start with greedy as a baseline
    greedy = _greedy_rows(n_rows, k)
    overall_best = [list(r) for r in greedy]
    overall_scope = _scope(overall_best)

    # Local search on greedy
    ls = _local_search([list(r) for r in greedy], n_rows, k, rng, deadline - 5)
    s = _scope(ls)
    if s < overall_scope:
        overall_best = ls
        overall_scope = s

    # Random restarts
    while time.time() < deadline - 10:
        # build random rows
        rows = []
        used = set()
        ok = True
        for _ in range(n_rows):
            # random row with growing max
            while True:
                row = [0]
                x = 1
                while len(row) < k + 1:
                    x += rng.randint(0, 3)
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        break
                if len(row) == k + 1:
                    break
            rows.append(row)
        s = _scope(rows)
        if s < overall_scope:
            overall_best = [list(r) for r in rows]
            overall_scope = s
        # local search on this
        ls = _local_search([list(r) for r in rows], n_rows, k, rng, deadline - 5)
        s2 = _scope(ls)
        if s2 < overall_scope:
            overall_best = ls
            overall_scope = s2
    return overall_best

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _DEADLINE
    rng = random.Random(12345)

    # Baseline: greedy
    greedy = _greedy_rows(n_rows, k)
    best = [list(r) for r in greedy]
    best_scope = _scope(best)

    # Local search on greedy
    ls = _local_search([list(r) for r in greedy], n_rows, k, rng, deadline - 30)
    s = _scope(ls)
    if s < best_scope:
        best = ls
        best_scope = s

    # Random restarts + local search
    while time.time() < deadline - 20:
        rows = []
        used = set()
        try:
            for _ in range(n_rows):
                row = [0]
                x = 1
                attempts = 0
                while len(row) < k + 1 and attempts < 100000:
                    x += rng.randint(0, 4)
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        attempts = 0
                    else:
                        attempts += 1
                if len(row) < k + 1:
                    raise ValueError
            rows.append(row)
            s = _scope(rows)
            if s < best_scope:
                best = [list(r) for r in rows]
                best_scope = s
            # local search
            ls = _local_search([list(r) for r in rows], n_rows, k, rng, deadline - 10)
            s2 = _scope(ls)
            if s2 < best_scope:
                best = ls
                best_scope = s2
        except Exception:
            pass

    # Final validation
    if not _is_valid(best):
        # fallback to greedy
        best = [list(r) for r in greedy]

    return best
