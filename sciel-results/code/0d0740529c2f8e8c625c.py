"""Difference triangle set (7,5) – interleaved greedy + SA + backtracking + polish.

Combines:
  * interleaved greedy initialization (balanced starting packing)
  * randomized interleaved restarts to find a smaller scope
  * simulated annealing with row-replacement moves (accepts worse temporarily)
  * a bounded backtracking search targeting best_scope - 1
  * a final deterministic polish pass (move entries downward)
"""
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 143.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _compute_used(rows):
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used


def _greedy_row(used, k, max_mark, rnd=None, skip_prob=0.0):
    """Greedy place one row of length k+1 starting at 0; smallest marks that keep
    all differences unused. Returns (row, local_diffs, ok)."""
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_mark:
            return None, local, False
        if skip_prob and rnd is not None and rnd.random() < skip_prob and x < max_mark:
            x += 1
            if x > max_mark:
                return None, local, False
        bad = False
        new = []
        for a in row:
            dv = x - a
            if dv in used or dv in local:
                bad = True
                break
            new.append(dv)
        if not bad:
            row.append(x)
            local.update(new)
    return row, local, True


def _interleaved_greedy(n_rows, k, max_mark, rnd=None, randomize=False):
    """Build rows interleaved: at each step, extend the row with the smallest
    current maximum (deterministic) or a random weighted pick (randomize)."""
    rows = [[0] for _ in range(n_rows)]
    used = set()

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        if randomize and rnd is not None:
            weights = [1.0 / (rows[i][-1] + 1) for i in candidates]
            total_w = sum(weights)
            r = rnd.random() * total_w
            cum = 0.0
            chosen = candidates[0]
            for idx, i in enumerate(candidates):
                cum += weights[idx]
                if r <= cum:
                    chosen = i
                    break
        else:
            chosen = min(candidates, key=lambda idx: rows[idx][-1])

        row = rows[chosen]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                found = True
                break
            x += 1
        if not found:
            return None, None

    for r in rows:
        if len(r) != k + 1:
            return None, None
    return rows, used


def _simulated_annealing(rows, n_rows, k, time_budget, rnd):
    """Simulated annealing: replace one row at a time with a newly built row that
    keeps the packing valid. Accepts worse scope with probability exp(-d/T)."""
    if rows is None:
        return None, None

    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope
    used = _compute_used(current)

    deadline = time.time() + time_budget
    T = 50.0
    cooling = 0.997

    while time.time() < deadline and T > 0.05:
        if _dead():
            break

        ri = rnd.randint(0, n_rows - 1)
        old_row = current[ri]
        old_diffs = _diffs_of_row(old_row)
        used_without = used - old_diffs

        max_mark = min(current_scope, best_scope) - 1
        if max_mark < k + 1:
            max_mark = best_scope

        new_row = None
        for attempt in range(15):
            skip = 0.05 + 0.1 * (attempt / 15.0)
            row, local, ok = _greedy_row(used_without, k, max_mark, rnd, skip_prob=skip)
            if ok and len(row) == k + 1:
                new_row = row
                break

        if new_row is None:
            T *= cooling
            continue

        new_diffs = _diffs_of_row(new_row)
        if new_diffs & used_without:
            T *= cooling
            continue

        others_scope = max(max(r) for i, r in enumerate(current) if i != ri)
        new_scope = max(others_scope, max(new_row))

        delta = new_scope - current_scope
        if delta <= 0 or rnd.random() < math.exp(-delta / T):
            used = used_without | new_diffs
            current[ri] = new_row
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [r[:] for r in current]
        T *= cooling

    return best, best_scope


def _backtrack_search(n_rows, k, target, time_budget=4.0):
    """Bounded backtracking: try to fill all rows within [0..target]."""
    rows = [None] * n_rows
    used = set()
    t0 = time.time()
    node_count = 0

    def fill_row(row_id, marks):
        nonlocal node_count
        if time.time() - t0 > time_budget:
            return False
        if len(marks) == k + 1:
            rows[row_id] = [0] + list(marks)
            return True
        start = (marks[-1] + 1) if marks else 1
        for x in range(start, target + 1):
            node_count += 1
            if node_count % 20000 == 0 and time.time() - t0 > time_budget:
                return False
            prev = [0] + list(marks)
            newl = set()
            bad = False
            for a in prev:
                dv = x - a
                if dv in used or dv in newl:
                    bad = True
                    break
                newl.add(dv)
            if bad:
                continue
            marks.append(x)
            used |= newl
            if fill_row(row_id + 1 if row_id + 1 < n_rows else row_id, marks):
                return True
            used -= newl
            marks.pop()
        return False

    # Fill rows one at a time, restarting marks each row
    def fill_all(row_id):
        if time.time() - t0 > time_budget:
            return False
        if row_id == n_rows:
            return True
        marks = []
        # try to build a full row
        def build_row():
            nonlocal node_count
            if len(marks) == k + 1:
                return True
            start = (marks[-1] + 1) if marks else 1
            for x in range(start, target + 1):
                node_count += 1
                if node_count % 20000 == 0 and time.time() - t0 > time_budget:
                    return False
                prev = [0] + list(marks)
                newl = set()
                bad = False
                for a in prev:
                    dv = x - a
                    if dv in used or dv in newl:
                        bad = True
                        break
                    newl.add(dv)
                if bad:
                    continue
                marks.append(x)
                used |= newl
                if build_row():
                    return True
                used -= newl
                marks.pop()
            return False

        if build_row():
            rows[row_id] = [0] + list(marks)
            if fill_all(row_id + 1):
                return True
            # undo
            for d in _diffs_of_row(rows[row_id]):
                used.discard(d)
            rows[row_id] = None
        return False

    if fill_all(0) and all(r is not None for r in rows):
        return rows
    return None


def _polish(rows, n_rows, k, time_budget):
    """Deterministic final polish: repeatedly try to move each entry (except 0)
    to the smallest valid smaller value. Accepts any valid move (keeps scope
    non-increasing)."""
    if rows is None:
        return None, None
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    used = _compute_used(best)
    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline and not _dead():
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            for pi in range(1, len(row)):
                old_val = row[pi]
                for new_val in range(old_val - 1, 0, -1):
                    if new_val <= row[pi - 1]:
                        break
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    old_diffs = set()
                    for a in row:
                        if a != old_val:
                            old_diffs.add(abs(old_val - a))
                    new_diffs = set()
                    for a in row:
                        if a != old_val:
                            new_diffs.add(abs(new_val - a))
                    remaining = used - old_diffs
                    if new_diffs & remaining:
                        continue
                    if 0 in new_diffs:
                        continue
                    used = remaining | new_diffs
                    row[pi] = new_val
                    improved = True
                    break
    return best, _scope(best)


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 143.0

    def elapsed():
        return time.time() - start_time

    rng = random.Random(12345)

    best_rows = None
    best_scope = float('inf')

    # --- Phase 1: deterministic interleaved greedy (baseline) ---
    rows, used = _interleaved_greedy(n_rows, k, 500)
    if rows:
        best_rows = rows
        best_scope = _scope(rows)

    if elapsed() > deadline - 3.0:
        return best_rows

    # --- Phase 2: randomized interleaved restarts ---
    for restart in range(200):
        if elapsed() > deadline - 25.0:
            break
        max_mark = best_scope - 1 if best_scope < float('inf') else 300
        rows, used = _interleaved_greedy(n_rows, k, max_mark, rnd=rng, randomize=True)
        if rows and all(len(r) == k + 1 for r in rows):
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = rows

    if elapsed() > deadline - 3.0:
        return best_rows

    # --- Phase 3: simulated annealing ---
    if best_rows is not None:
        remaining = deadline - time.time() - 20.0
        if remaining > 3.0:
            improved_rows, improved_scope = _simulated_annealing(
                best_rows, n_rows, k, remaining, rng)
            if improved_rows is not None and improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows

    if elapsed() > deadline - 3.0:
        return best_rows

    # --- Phase 4: bounded backtracking targeting best_scope - 1 ---
    if best_scope < float('inf') and elapsed() < deadline - 12.0:
        target = best_scope - 1
        bt = _backtrack_search(n_rows, k, target, time_budget=min(6.0, deadline - time.time() - 8.0))
        if bt is not None:
            sc = _scope(bt)
            if sc < best_scope:
                best_scope = sc
                best_rows = bt

    if elapsed() > deadline - 3.0:
        return best_rows

    # --- Phase 5: final polish pass ---
    if best_rows is not None:
        remaining = deadline - time.time() - 2.0
        if remaining > 0.5:
            polished, polished_scope = _polish(best_rows, n_rows, k, remaining)
            if polished is not None and polished_scope < best_scope:
                best_scope = polished_scope
                best_rows = polished

    return best_rows
