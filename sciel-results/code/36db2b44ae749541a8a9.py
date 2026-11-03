import time
import random
import math
import sys
from collections import Counter, defaultdict

_DEADLINE = 0.0
_START = 0.0


def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])


def _valid(rows, k):
    if len(rows) != k + 1:
        pass
    seen = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True


def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts


def _row_diffs(row):
    """Return list of all differences in a row."""
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs


def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]


def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1


def _collisions(diff_counts):
    return sum(max(0, c - 1) for c in diff_counts.values())


def _cost(rows, diff_counts):
    col = _collisions(diff_counts)
    if col > 0:
        return 10000 + col
    return _scope(rows)


def _greedy_row(k, used_diffs, rng, max_scope):
    """Greedily build a row avoiding used_diffs. Returns list or None."""
    row = [0]
    internal = set()
    for step in range(1, k + 1):
        found = False
        for t in range(1, max_scope + 1):
            ok = True
            for a in row:
                d = t - a
                if d <= 0:
                    continue
                if d in used_diffs or d in internal:
                    ok = False
                    break
            if ok:
                row.append(t)
                internal.add(t - 0)  # diff from 0
                for a in row[:-1]:
                    internal.add(t - a)
                found = True
                break
        if not found:
            return None
    return row


def _csp_build_row(k, used_diffs, rng, max_scope):
    """
    Backtracking with forward checking.
    Builds a row of k+1 elements starting with 0.
    """
    row = [0]
    internal = set()

    def backtrack(pos, current_max):
        if pos == k:
            return True
        # Try candidates in randomized order
        candidates = list(range(current_max + 1, max_scope + 1))
        rng.shuffle(candidates)
        for t in candidates:
            # Check all differences with existing elements
            conflict = False
            new_diffs = []
            for a in row:
                d = t - a
                if d <= 0:
                    continue
                if d in used_diffs or d in internal:
                    conflict = True
                    break
                new_diffs.append(d)
            if conflict:
                continue
            # Forward check: can we still place remaining elements?
            # Simple: just check if there's enough room
            remaining = k - pos - 1
            if remaining > 0:
                # We need at least 'remaining' more values after t
                # each at least 1 apart, so we need t + remaining <= max_scope
                # (very loose check)
                if t + remaining > max_scope + 2 * remaining:
                    continue

            row.append(t)
            for d in new_diffs:
                internal.add(d)

            if backtrack(pos + 1, t):
                return True

            row.pop()
            for d in new_diffs:
                internal.discard(d)

        return False

    if backtrack(0, 0):
        return list(row)
    return None


def _sa_optimize(rows, k, deadline, rng):
    """
    Simulated annealing to optimize the row set.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    diff_counts = _build_diff_counts(current)
    cur_cost = _cost(current, diff_counts)

    best = [list(r) for r in current]
    best_cost = cur_cost
    best_scope = _scope(best)

    T = 20.0
    min_T = 0.5
    cooling = 0.9995

    # Pre-compute max element position
    def find_max_row_pos():
        mx = 0
        ri = -1
        pi = -1
        for i in range(n_rows):
            for j in range(1, len(current[i])):
                if current[i][j] > mx:
                    mx = current[i][j]
                    ri = i
                    pi = j
        return ri, pi, mx

    it = 0
    while time.time() < deadline:
        if it % 5000 == 0:
            T = max(min_T, T * cooling)

        r = rng.random()

        if r < 0.15:
            # --- Row rebuild ---
            i = rng.randint(0, n_rows - 1)
            old_row = list(current[i])
            _remove_row_diffs(diff_counts, old_row)

            new_row = _csp_build_row(k, diff_counts, rng, best_scope + 15)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                it += 1
                continue

            _add_row_diffs(diff_counts, new_row)
            current[i] = new_row
            nc = _cost(current, diff_counts)

            if nc < cur_cost or rng.random() < math.exp(-(nc - cur_cost) / max(T, 0.01)):
                cur_cost = nc
                if nc < best_cost:
                    best_cost = nc
                    best_scope = _scope(current)
                    best = [list(x) for x in current]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current[i] = old_row

        elif r < 0.30:
            # --- Targeted scope reduction on the max element ---
            ri, pi, mx = find_max_row_pos()
            if ri < 0:
                it += 1
                continue
            i = ri
            j = pi
            old_val = current[i][j]
            prev_val = current[i][j - 1]
            if old_val - prev_val <= 1:
                it += 1
                continue
            new_val = rng.randint(prev_val + 1, old_val - 1)
            if new_val == old_val:
                it += 1
                continue

            _remove_row_diffs(diff_counts, current[i])
            current[i][j] = new_val
            _add_row_diffs(diff_counts, current[i])
            nc = _cost(current, diff_counts)

            if nc < cur_cost or rng.random() < math.exp(-(nc - cur_cost) / max(T, 0.01)):
                cur_cost = nc
                if nc < best_cost:
                    best_cost = nc
                    best_scope = _scope(current)
                    best = [list(x) for x in current]
            else:
                _remove_row_diffs(diff_counts, current[i])
                current[i][j] = old_val
                _add_row_diffs(diff_counts, current[i])

        else:
            # --- Single-element move ---
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = current[i][j]
            lo = current[i][j - 1] + 1
            hi = (current[i][j + 1] - 1) if j < k else (best_scope + 5)

            if lo > hi:
                it += 1
                continue

            # Gaussian step
            span = hi - lo
            if span <= 0:
                it += 1
                continue
            delta = int(rng.gauss(0, max(1, span * 0.25)))
            new_val = old_val + delta
            if new_val < lo:
                new_val = lo
            elif new_val > hi:
                new_val = hi
            if new_val == old_val:
                it += 1
                continue

            _remove_row_diffs(diff_counts, current[i])
            current[i][j] = new_val
            _add_row_diffs(diff_counts, current[i])
            nc = _cost(current, diff_counts)

            if nc < cur_cost or rng.random() < math.exp(-(nc - cur_cost) / max(T, 0.01)):
                cur_cost = nc
                if nc < best_cost:
                    best_cost = nc
                    best_scope = _scope(current)
                    best = [list(x) for x in current]
            else:
                _remove_row_diffs(diff_counts, current[i])
                current[i][j] = old_val
                _add_row_diffs(diff_counts, current[i])

        it += 1

    return best


def construct(n_rows, k):
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = _START + 115.0

    rng = random.Random(42)

    # --- Phase 1: Build diverse initial candidates ---
    candidates = []

    # Candidate A: Greedy with tight packing
    rowsA = []
    usedA = set()
    for _ in range(n_rows):
        row = _greedy_row(k, usedA, rng, 120)
        if row is None:
            row = _csp_build_row(k, usedA, rng, 130)
        if row is None:
            row = [0, 1, 2, 3, 4, 5]
        rowsA.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                usedA.add(row[j] - row[i])
    candidates.append(rowsA)

    # Candidate B: Greedy with wider spread
    rowsB = []
    usedB = set()
    for _ in range(n_rows):
        row = _csp_build_row(k, usedB, rng, 130)
        if row is None:
            row = _greedy_row(k, usedB, rng, 140)
        if row is None:
            row = [0, 3, 6, 9, 12, 15]
        rowsB.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                usedB.add(row[j] - row[i])
    candidates.append(rowsB)

    # Candidate C: Another CSP with different seed
    rng2 = random.Random(99)
    rowsC = []
    usedC = set()
    for _ in range(n_rows):
        row = _csp_build_row(k, usedC, rng2, 125)
        if row is None:
            row = _greedy_row(k, usedC, rng2, 135)
        if row is None:
            row = [0, 2, 5, 9, 14, 20]
        rowsC.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                usedC.add(row[j] - row[i])
    candidates.append(rowsC)

    # Sort by scope
    candidates.sort(key=lambda r: _scope(r))

    best_rows = None
    best_scope = float('inf')

    # --- Phase 2: SA optimization on top candidates ---
    # Allocate time: 40% to best, 30% to second, 30% to polish
    t1 = _START + 46.0
    t2 = _START + 80.0
    t3 = _DEADLINE - 2.0

    if len(candidates) > 0 and time.time() < t1:
        res = _sa_optimize(candidates[0], k, t1, rng)
        if _valid(res, k):
            s = _scope(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    if len(candidates) > 1 and time.time() < t2:
        res = _sa_optimize(candidates[1], k, t2, rng)
        if _valid(res, k):
            s = _scope(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    if len(candidates) > 2 and time.time() < t2 + 10:
        res = _sa_optimize(candidates[2], k, min(t2 + 10, _DEADLINE - 5), rng)
        if _valid(res, k):
            s = _scope(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # --- Phase 3: Polish the best found ---
    if best_rows is not None and time.time() < t3:
        res = _sa_optimize(best_rows, k, t3, rng)
        if _valid(res, k):
            s = _scope(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    if best_rows is not None:
        return best_rows

    # Fallback
    return candidates[0]
