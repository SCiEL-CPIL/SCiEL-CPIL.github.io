"""Difference triangle set (4,7): minimize the scope (largest entry)."""
import time
import random
import numpy as np

_DEADLINE = 140.0
_START = time.time()


def _diff_mask(row):
    n = len(row)
    m = np.zeros(n * (n - 1) // 2, dtype=bool)
    r = 0
    for j in range(1, n):
        for i in range(j):
            m[r] = (row[j] - row[i])
            r += 1
    return m


def _valid(rows):
    n = len(rows[0])
    cap = 100000
    used = np.zeros(cap, dtype=bool)
    for row in rows:
        m = _diff_mask(row)
        if m.max() >= cap:
            return False
        used[m] = True
        m = _diff_mask(row)
        if m.max() >= cap:
            return False
        if np.count_nonzero(used[m]) != m.sum():
            return False
    return True


def _construct_greedy(n_rows, k, rng):
    n = k + 1
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for pos in range(1, n):
            rem = n - pos
            lo = row[-1] + 1
            hi = row[-1] + rem + 25
            best_val = None
            best_score = None
            for x in range(lo, hi + 1):
                nd = {x - a for a in row}
                if nd & used:
                    continue
                if rem == 1:
                    score = x
                else:
                    score = x + rem
                if best_score is None or score < best_score:
                    best_score = score
                    best_val = x
                    if rem == 1:
                        break
            if best_val is None:
                x = row[-1] + 1
                while True:
                    nd = {x - a for a in row}
                    if not (nd & used):
                        best_val = x
                        break
                    x += 1
            row.append(best_val)
            used |= {best_val - a for a in row[:-1]}
        rows.append(row)
    return rows


def _targeted_descent(rows, rng):
    n = len(rows[0])
    order = list(range(len(rows)))
    improved = True
    while improved:
        improved = False
        rows.sort(key=lambda r: -r[-1])
        for ri in order:
            row = rows[ri]
            max_val = row[-1]
            for new_last in range(max_val - 1, row[-2]):
                trial = row[:-1] + [new_last]
                nd = {trial[-1] - a for a in trial[:-1]}
                conflict = None
                for oj, other in enumerate(rows):
                    if oj == ri:
                        continue
                    od = {other[b] - other[a] for a in range(n) for b in range(a + 1, n)}
                    hit = nd & od
                    if hit:
                        conflict = (oj, hit)
                        break
                if conflict is None:
                    rows[ri] = trial
                    improved = True
                    break
                oj, hit = conflict
                other = rows[oj]
                moved = False
                for d in hit:
                    for (a, b) in [(i, j) for j in range(1, n) for i in range(j) if other[j] - other[i] == d]:
                        cand = other[b] - d
                        if 0 < cand < other[b] and cand not in other:
                            other[b] = cand
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    rows[ri] = trial
                    improved = True
                    break
    return rows


def construct(n_rows, k):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0

    best = None
    best_scope = None
    rng = random.Random(12345)

    def scope_of(rows):
        return max(r[-1] for r in rows)

    # Initial construction
    cand = _construct_greedy(n_rows, k, rng)
    if _valid(cand):
        cand = _targeted_descent(cand, rng)
        if _valid(cand):
            best = cand
            best_scope = scope_of(cand)

    # Randomized restarts
    while time.time() < _DEADLINE - 2.0:
        cand = _construct_greedy(n_rows, k, rng)
        if not _valid(cand):
            continue
        cand = _targeted_descent(cand, rng)
        if _valid(cand):
            s = scope_of(cand)
            if best_scope is None or s < best_scope:
                best = cand
                best_scope = s

    if best is None:
        # Fallback: pure greedy (always valid)
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        return rows
    return best
