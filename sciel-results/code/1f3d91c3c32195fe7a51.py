"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

Structural approach:
  * Parallel greedy construction that keeps row scopes balanced (structural core).
  * Backtracking row rebuild under a strict upper bound (refinement machinery).
  * Collision-driven local search with a simulated-annealing cooling schedule
    over a global difference-collision view (max-color index = scope).
"""
import time
import random

_TIME_LIMIT = 120.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        prev = -1
        for v in row:
            if v <= prev:
                return False
            prev = v
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(r[-1] for r in rows)


def _row_diffs(row):
    n = len(row)
    return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}


def _all_diffs_except(rows, idx):
    used = set()
    for i, row in enumerate(rows):
        if i == idx:
            continue
        used |= _row_diffs(row)
    return used


def _construct_row_partial(k, used, max_val, rng, mode):
    """Build one full row (k+1 entries, starts at 0, increasing, all <= max_val)
    with no difference in `used`. Returns the row or None on failure."""
    row = [0]
    target = k + 1
    while len(row) < target:
        last = row[-1]
        candidates = []
        for cand in range(last + 1, max_val + 1):
            nd = {cand - a for a in row}
            if not (nd & used):
                candidates.append(cand)
        if not candidates:
            return None
        if mode == "small":
            x = candidates[0]
        elif mode == "flex":
            best = candidates[0]
            best_score = -1
            for cand in candidates[:12]:
                nd = {cand - a for a in row}
                cnt = 0
                for nxt in range(cand + 1, min(cand + 20, max_val) + 1):
                    nd2 = {nxt - a for a in row}
                    nd2.add(nxt - cand)
                    if not (nd2 & (used | nd)):
                        cnt += 1
                        if cnt >= 4:
                            break
                score = cnt * 100 - cand
                if score > best_score:
                    best_score = score
                    best = cand
            x = best
        else:  # "rand"
            pool = candidates[:min(8, len(candidates))]
            x = rng.choice(pool)
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row


def _backtrack_row(used, k, max_scope, rng, max_nodes=20000):
    """Build one row (k+1 entries, starts at 0, increasing, all <= max_scope)
    with no difference in `used`, using backtracking. Returns row or None."""
    row = [0]
    target = k + 1
    nodes = [0]

    def bt():
        if len(row) == target:
            return True
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return False
        last = row[-1]
        lo = last + 1
        hi = max_scope
        # forward check: need target-len(row) more entries strictly increasing
        # so last entry at least hi must allow room; keep simple bound
        start = lo
        # candidate order: small first with slight randomization for diversity
        cands = list(range(start, hi + 1))
        if rng is not None and len(cands) > 8:
            # shuffle the tail to add diversity but keep the front ordered
            tail = cands[8:]
            rng.shuffle(tail)
            cands = cands[:8] + tail
        for x in cands:
            # feasibility: remaining slots must fit in [x+1 .. max_scope]
            need = target - len(row) - 1
            if max_scope - x < need:
                # x too large? actually x large leaves less room -> break
                break
            nd = {x - a for a in row}
            if nd & used:
                continue
            row.append(x)
            used |= nd
            if bt():
                return True
            row.pop()
            used.difference_update(nd)
        return False

    return row[:] if bt() else None


def _construct_parallel(n_rows, k, rng, mode, scope_budget=None):
    """Build all rows in parallel, keeping scopes balanced."""
    rows = [[0] for _ in range(n_rows)]
    used = set()
    target = k + 1
    while any(len(r) < target for r in rows):
        order = sorted(range(n_rows), key=lambda i: (len(rows[i]), rows[i][-1]))
        for i in order:
            if len(rows[i]) >= target:
                continue
            if scope_budget is not None:
                max_val = scope_budget
                row = rows[i]
                need = target - len(row)
                if max_val - row[-1] < need:
                    return None
                new_row = _construct_row_partial(k, used, max_val, rng, mode)
                if new_row is None:
                    return None
                rows[i] = new_row
                used |= _row_diffs(new_row)
            else:
                last = rows[i][-1]
                chosen = None
                for cand in range(last + 1, last + 400):
                    nd = {cand - a for a in rows[i]}
                    if not (nd & used):
                        chosen = cand
                        break
                if chosen is None:
                    return None
                rows[i] = rows[i] + [chosen]
                used |= {chosen - a for a in rows[i]}
    return rows


def _rebuild_row(row_idx, rows, k, max_val, rng, max_nodes=20000):
    """Rebuild rows[row_idx] to stay <= max_val without colliding with the
    other rows' differences. Returns new row or None."""
    other_used = _all_diffs_except(rows, row_idx)
    return _backtrack_row(other_used, k, max_val, rng, max_nodes)


def _local_search(rows, n_rows, k, rng, max_iters=3000):
    """Collision-driven local search with a simulated-annealing cooling schedule.
    Energy = scope (max entry). Each move tries to reduce the largest-scope row
    by rebuilding it under a tighter bound (deterministic improvement), and on
    failure nudges an interior element to free space. A temperature controls how
    aggressively we accept temporarily non-improving interior nudges."""
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    T = float(best_scope) * 0.25
    T_min = 1.0
    cool = 0.985

    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        # target the largest-scope row
        i = max(range(n_rows), key=lambda j: rows[j][-1])
        cur_scope = rows[i][-1]
        found = False

        # 1) try to reduce last element via rebuild under tighter bound
        for bound in range(cur_scope - 1, max(cur_scope - 12, rows[i][-2]), -1):
            if bound <= rows[i][-2]:
                break
            new_row = _rebuild_row(i, rows, k, bound, rng)
            if new_row is not None:
                rows[i] = new_row
                found = True
                break
        if found:
            if _scope(rows) < best_scope:
                best_scope = _scope(rows)
            T = max(T_min, T * cool)
            continue

        # 2) nudge an interior element to open space (annealed acceptance)
        row = rows[i]
        other_used = _all_diffs_except(rows, i)
        improved = False
        for pos in range(len(row) - 2, 0, -1):
            cur = row[pos]
            for new_val in range(cur - 1, row[pos - 1], -1):
                if new_val >= row[pos + 1]:
                    continue
                candidate_row = row[:]
                candidate_row[pos] = new_val
                new_d = _row_diffs(candidate_row)
                if new_d & other_used:
                    continue
                cand_scope = max(r[-1] for r in rows[:i] + [candidate_row] + rows[i + 1:])
                if cand_scope <= best_scope:
                    rows[i] = candidate_row
                    improved = True
                    if cand_scope < best_scope:
                        best_scope = cand_scope
                    break
            if improved:
                break
        if improved:
            T = max(T_min, T * cool)
        else:
            # no improving move: small probability of a neutral/repairing shuffle
            # to escape local minima (rare, keeps validity always)
            if T > T_min and rng.random() < 0.02:
                # try a random valid interior lowering on the target row
                pos = rng.randint(1, len(row) - 2)
                cur = row[pos]
                for new_val in range(cur - 1, row[pos - 1], -1):
                    if new_val >= row[pos + 1]:
                        continue
                    candidate_row = row[:]
                    candidate_row[pos] = new_val
                    if _row_diffs(candidate_row) & other_used:
                        continue
                    rows[i] = candidate_row
                    break
            T = max(T_min, T * cool)

    return rows


def _iterative_tightening(rows, n_rows, k, rng, max_rounds=40):
    """Repeatedly rebuild each row under a tighter bound than its current scope."""
    rows = [list(r) for r in rows]
    for _ in range(max_rounds):
        if _time_left() < 2.0:
            break
        improved = False
        order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
        for i in order:
            if _time_left() < 2.0:
                break
            cur_scope = rows[i][-1]
            for bound in range(cur_scope - 1, max(cur_scope - 12, rows[i][-2]), -1):
                if bound <= rows[i][-2]:
                    break
                new_row = _rebuild_row(i, rows, k, bound, rng)
                if new_row is not None:
                    rows[i] = new_row
                    improved = True
                    break
        if not improved:
            break
    return rows


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)

    # Baseline: parallel greedy (natural, balanced)
    best_rows = _construct_parallel(n_rows, k, rng, "flex")
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)
    best_scope = _scope(best_rows)

    # Multi-strategy restarts
    modes = ["flex", "small", "rand"]
    for attempt in range(40):
        if _time_left() < 5.0:
            break
        mode = modes[attempt % len(modes)]
        rows = _construct_parallel(n_rows, k, rng, mode)
        if rows is None:
            continue
        cur = _scope(rows)
        if cur < best_scope:
            best_scope = cur
            best_rows = rows
        if _time_left() > 5.0:
            tightened = _iterative_tightening(rows, n_rows, k, rng, max_rounds=15)
            tc = _scope(tightened)
            if tc < best_scope:
                best_scope = tc
                best_rows = tightened

    # Final aggressive local search (annealed) + tightening
    if _time_left() > 3.0:
        ls = _local_search(best_rows, n_rows, k, rng, max_iters=3000)
        if _scope(ls) < best_scope:
            best_rows = ls
            best_scope = _scope(ls)
        if _time_left() > 3.0:
            tightened = _iterative_tightening(best_rows, n_rows, k, rng, max_rounds=30)
            if _scope(tightened) < best_scope:
                best_rows = tightened
                best_scope = _scope(tightened)
        if _time_left() > 3.0:
            ls2 = _local_search(best_rows, n_rows, k, rng, max_iters=1500)
            if _scope(ls2) < best_scope:
                best_rows = ls2
                best_scope = _scope(ls2)

    if not _validate(best_rows, n_rows, k):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)

    return best_rows
