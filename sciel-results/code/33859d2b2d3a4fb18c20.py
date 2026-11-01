import time
import random

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _row_diffs(row):
    """Set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _scope(rows):
    return max(max(r) for r in rows)


def _is_valid(rows, n_rows, k):
    seen = set()
    for row in rows:
        d = _row_diffs(row)
        if d & seen:
            return False
        seen |= d
    return True


def _greedy_build(n_rows, k, rng, order=None):
    """Greedy: build rows in given order."""
    if order is None:
        order = list(range(n_rows))
        rng.shuffle(order)
    used = set()
    rows = [None] * n_rows
    for r in order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows


def _targeted_descent(rows, row_diffs, all_diffs, n_rows, k, rng, budget):
    """
    Descent pass: try to reduce the scope by shrinking the element that holds
    the current global max (the last element of the max row), then other elements.
    Uses incremental diff bookkeeping. Returns (rows, row_diffs, all_diffs, scope).
    """
    start = time.time()
    cur = [list(r) for r in rows]
    rd = [set(s) for s in row_diffs]
    ad = set(all_diffs)
    cur_scope = _scope(cur)
    no_improve = 0
    max_no_improve = 400

    while time.time() - start < budget and _time_left() > 2.0:
        if no_improve > max_no_improve:
            break

        # Find rows holding the current global max
        max_val = max(max(r) for r in cur)
        max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
        r = rng.choice(max_rows)
        row = cur[r]

        # 70%: reduce the last element (the max of this row)
        if rng.random() < 0.7:
            pos = k
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if old_val <= lower:
                no_improve += 1
                continue
            # candidate new values: sequential near the bottom (best reduction) + samples
            cands = set()
            for v in range(lower, min(lower + 12, old_val)):
                cands.add(v)
            for _ in range(8):
                cands.add(rng.randint(lower, old_val - 1))
            cands = sorted(cands, reverse=True)  # try larger reductions first
        else:
            # reduce a non-last element in the max row
            if k < 2:
                no_improve += 1
                continue
            pos = rng.randint(1, k - 1)
            old_val = row[pos]
            lo2 = row[pos - 1] + 1
            hi2 = row[pos + 1] - 1
            if hi2 <= lo2:
                no_improve += 1
                continue
            cands = []
            for _ in range(8):
                cands.append(rng.randint(lo2, hi2))
            cands = list(set(cands))

        moved = False
        for new_val in cands:
            if new_val <= row[pos - 1] or (pos < k and new_val >= row[pos + 1]):
                continue
            row[pos] = new_val
            new_rd = _row_diffs(row)
            if len(new_rd) != k * (k + 1) // 2:
                row[pos] = old_val
                continue
            other_diffs = ad - rd[r]
            if new_rd & other_diffs:
                row[pos] = old_val
                continue
            # accept: max only decreases or stays; if it stays, keep to diversify
            ad = (ad - rd[r]) | new_rd
            rd[r] = new_rd
            new_scope = _scope(cur)
            if new_scope < cur_scope:
                cur_scope = new_scope
            moved = True
            break

        if moved:
            no_improve = 0
        else:
            no_improve += 1

    return cur, rd, ad, cur_scope


def _anneal(cur, row_diffs, all_diffs, n_rows, k, rng, T0, steps, deadline):
    """
    Simulated annealing with incremental bookkeeping.
    Accepts only valid (conflict-free) moves; Metropolis on the scope delta.
    Returns (cur, row_diffs, all_diffs, cur_scope).
    """
    start = time.time()
    T = T0
    cur_scope = _scope(cur)
    for _ in range(steps):
        if time.time() > deadline or _time_left() < 2.0:
            break
        ri = rng.randrange(n_rows)
        pos = rng.randint(1, k)
        row = cur[ri]
        old_val = row[pos]
        lo = row[pos - 1] + 1
        hi = row[pos + 1] - 1 if pos < k else cur_scope + 20
        if hi <= lo:
            continue
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue

        row[pos] = new_val
        new_rd = _row_diffs(row)
        if len(new_rd) != k * (k + 1) // 2:
            row[pos] = old_val
            T *= 0.99995
            continue
        other_diffs = all_diffs - row_diffs[ri]
        if new_rd & other_diffs:
            row[pos] = old_val
            T *= 0.99995
            continue

        # valid move
        new_scope = _scope(cur)
        delta = new_scope - cur_scope
        if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
            all_diffs = (all_diffs - row_diffs[ri]) | new_rd
            row_diffs[ri] = new_rd
            cur_scope = new_scope
        else:
            row[pos] = old_val
        T *= 0.99995
    return cur, row_diffs, all_diffs, cur_scope


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + _DEADLINE

    rng = random.Random(42)

    # Phase 1: multiple greedy starts
    best_rows = None
    best_max = float('inf')
    for i in range(15):
        if _time_left() < 15.0:
            break
        rows = _greedy_build(n_rows, k, rng)
        cur_max = _scope(rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = _scope(best_rows)

    # bookkeeping for best
    best_rd = [_row_diffs(r) for r in best_rows]
    best_ad = set()
    for rd in best_rd:
        best_ad |= rd

    # Phase 2: ILS (anneal + kick + targeted descent)
    max_restarts = 400
    for restart in range(max_restarts):
        if _time_left() < 8.0:
            break

        # start from best, then kick (perturb) for restarts > 0
        cur = [list(r) for r in best_rows]
        if restart > 0:
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, k)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < k else best_max + 10
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        rd = [_row_diffs(r) for r in cur]
        ad = set()
        for s in rd:
            ad |= s
        cur_scope = _scope(cur)

        if cur_scope < best_max:
            best_max = cur_scope
            best_rows = [list(r) for r in cur]
            best_rd = [set(s) for s in rd]
            best_ad = set(ad)

        # anneal
        T0 = max(0.05, 1.0 - restart / max_restarts)
        steps = 40000
        cur, rd, ad, cur_scope = _anneal(cur, rd, ad, n_rows, k, rng, T0, steps, deadline)

        if cur_scope < best_max:
            best_max = cur_scope
            best_rows = [list(r) for r in cur]
            best_rd = [set(s) for s in rd]
            best_ad = set(ad)

        # targeted descent to squeeze scope
        if _time_left() > 6.0:
            cur, rd, ad, cur_scope = _targeted_descent(
                cur, rd, ad, n_rows, k, rng, budget=1.5)
            if cur_scope < best_max:
                best_max = cur_scope
                best_rows = [list(r) for r in cur]
                best_rd = [set(s) for s in rd]
                best_ad = set(ad)

    # Final verification; fallback to greedy if anything broke
    if not _is_valid(best_rows, n_rows, k):
        best_rows = _greedy_build(n_rows, k, rng)

    return best_rows
