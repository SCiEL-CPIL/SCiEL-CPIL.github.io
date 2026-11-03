import time
import random
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0


def _fft_conflict(diffs, universe):
    """Return the number of difference values that occur more than once.
    diffs is a flat array of positive integers (each row's internal differences).
    A conflict count of 0 means every difference is globally distinct."""
    if len(diffs) == 0:
        return 0
    d = np.asarray(diffs, dtype=np.int64)
    if d.max() > universe:
        return 10**9  # out of range -> treat as conflict
    a = np.zeros(universe + 1, dtype=np.int64)
    a[d] = 1
    c = np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(a))
    c = c[:universe + 1]
    return int((c > 1).sum())


def _complete_row(used_set, x, k, universe, rng):
    """Complete the current row from its last element x.
    Returns (row, new_used_set, success)."""
    row = [0] + [x]
    used = set(used_set)
    used.add(x)
    for _ in range(k - 1):
        # collect candidate next marks
        cands = []
        lo = x + 1
        hi = min(universe, x + 400)
        for v in range(lo, hi + 1):
            bad = False
            for a in row:
                if (v - a) in used:
                    bad = True
                    break
            if not bad:
                cands.append(v)
        if not cands:
            return row, used, False
        if len(cands) == 1:
            v = cands[0]
        else:
            # heuristic: prefer the smallest, but occasionally jump to diversify
            if rng.random() < 0.7:
                v = cands[0]
            else:
                v = cands[rng.randrange(len(cands))]
        row.append(v)
        used.add(v)
        x = v
    return row, used, True


def _search_one(k, universe, start_first, rng):
    """One randomized backtracking search for a valid set of rows.
    Returns (rows, max_entry) on success else (None, None)."""
    used_set = set()
    rows = []
    # first element of each row (index 1)
    firsts = []
    cur_row = [0]

    def backtrack(depth, x):
        if depth == 0:
            # complete this row
            row, used_new, ok = _complete_row(used_set, x, k, universe, rng)
            if not ok:
                return False
            rows.append(row)
            return True
        # choose the first element of the next row
        lo = start_first + depth
        hi = min(universe, lo + 300)
        order = list(range(lo, hi + 1))
        rng.shuffle(order)
        for v in order:
            if (v) in used_set:
                continue
            # quick feasibility: v must not collide with any existing diff
            # (v - 0 = v) already checked; also v - a for a in rows? no,
            # differences are only within a row, so only v itself matters here
            firsts.append(v)
            used_set.add(v)
            cur_row.append(v)
            if backtrack(depth - 1, v):
                return True
            cur_row.pop()
            used_set.discard(v)
            firsts.pop()
        return False

    # seed the first row's first element
    firsts.append(start_first)
    used_set.add(start_first)
    cur_row.append(start_first)
    ok = backtrack(k, start_first)
    if ok:
        return rows, max(r[-1] for r in rows)
    return None, None


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN

    # The number of internal differences per row is k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    # counting bound: need total_diffs <= universe, universe ~ max entry
    count_bound = total_diffs
    # best known is 134 for (4,7); allow a generous universe
    base_universe = max(count_bound + 10, 150)

    rng = random.Random(12345)

    best_rows = None
    best_max = None

    # Try a range of universes and start values
    universe = base_universe
    start_first = 1
    attempts = 0
    while time.time() < deadline:
        rows, mx = _search_one(k, universe, start_first, rng)
        attempts += 1
        if rows is not None:
            if best_max is None or mx < best_max:
                best_max = mx
                best_rows = [list(r) for r in rows]
                # tighten universe to just above found max and try to improve
                universe = mx + 8
                if time.time() < deadline:
                    continue
        # vary start and universe to diversify
        start_first = rng.randint(1, max(1, universe // 3))
        if attempts % 5 == 0:
            universe = max(count_bound + 10, base_universe + rng.randint(0, 30))

    if best_rows is None:
        # Fallback: pure greedy (always valid)
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
    return best_rows
