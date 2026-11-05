"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

Combined approach:
  * Bitset-based constraint tracking for speed (from inspiration).
  * Parallel greedy construction with balanced scopes (from current).
  * Simulated annealing to escape local minima (from lessons).
  * Iterative scope-tightening and collision-driven local repair (from current).
"""
import time
import random

_DEADLINE = None
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
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
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}


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
            for cand in candidates[:15]:
                nd = {cand - a for a in row}
                cnt = 0
                for nxt in range(cand + 1, min(cand + 25, max_val) + 1):
                    nd2 = {nxt - a for a in row}
                    nd2.add(nxt - cand)
                    if not (nd2 & (used | nd)):
                        cnt += 1
                        if cnt >= 5:
                            break
                score = cnt * 100 - cand
                if score > best_score:
                    best_score = score
                    best = cand
            x = best
        else:  # "rand"
            pool = candidates[:min(10, len(candidates))]
            x = rng.choice(pool)
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row


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
                row = _construct_row_partial(k, used, max_val, rng, mode)
                if row is None:
                    return None
            else:
                last = rows[i][-1]
                chosen = None
                for cand in range(last + 1, last + 500):
                    nd = {cand - a for a in rows[i]}
                    if not (nd & used):
                        chosen = cand
                        break
                if chosen is None:
                    return None
                row = rows[i] + [chosen]
                used |= {chosen - a for a in rows[i]}
            rows[i] = row
            used |= _row_diffs(row)
    return rows


def _rebuild_row(row_idx, rows, k, used, max_val, rng):
    """Rebuild rows[row_idx] to stay <= max_val without colliding with `used`."""
    other_used = _all_diffs_except(rows, row_idx)
    new_row = _construct_row_partial(k, other_used, max_val, rng, "flex")
    return new_row


def _simulated_annealing(rows, n_rows, k, rng, max_iters=300, initial_temp=50.0, cooling=0.95):
    """SA to reduce the largest entry by perturbing interior elements."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    temp = initial_temp
    for _ in range(max_iters):
        if _time_left() < 2.0:
            break
        # Pick a random row and a random interior position
        i = rng.randrange(n_rows)
        row = rows[i]
        if len(row) <= 2:
            continue
        pos = rng.randrange(1, len(row) - 1)
        cur = row[pos]
        # Perturb: try to move the element
        delta = rng.choice([-1, 1]) * rng.randint(1, 5)
        new_val = cur + delta
        # Check bounds
        if new_val <= row[pos - 1] or new_val >= row[pos + 1]:
            continue
        other_used = _all_diffs_except(rows, i)
        candidate_row = row[:]
        candidate_row[pos] = new_val
        new_d = _row_diffs(candidate_row)
        if new_d & other_used:
            continue
        # Compute new scope
        new_scope = _scope(rows) if i != max(range(n_rows), key=lambda j: rows[j][-1]) else new_val
        # Accept if better or with probability
        if new_scope < best_scope:
            rows[i] = candidate_row
            best_rows = [list(r) for r in rows]
            best_scope = new_scope
        else:
            # Metropolis criterion
            if rng.random() < 0.1:
                rows[i] = candidate_row
        temp *= cooling
    return best_rows


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
            for bound in range(cur_scope - 1, max(cur_scope - 15, rows[i][-2]), -1):
                if bound <= rows[i][-2]:
                    break
                new_row = _rebuild_row(i, rows, k, None, bound, rng)
                if new_row is not None:
                    rows[i] = new_row
                    improved = True
                    break
        if not improved:
            break
    return rows


def _local_search(rows, n_rows, k, rng, max_iters=400):
    """Collision-driven repair: repeatedly try to shrink the largest-scope row."""
    rows = [list(r) for r in rows]
    for _ in range(max_iters):
        if _time_left() < 2.0:
            break
        i = max(range(n_rows), key=lambda j: rows[j][-1])
        cur_scope = rows[i][-1]
        found = False
        for bound in range(cur_scope - 1, max(cur_scope - 12, rows[i][-2]), -1):
            new_row = _rebuild_row(i, rows, k, None, bound, rng)
            if new_row is not None:
                rows[i] = new_row
                found = True
                break
        if found:
            continue
        row = rows[i]
        for pos in range(len(row) - 2, 0, -1):
            cur = row[pos]
            for new_val in range(cur - 1, row[pos - 1], -1):
                if new_val >= row[pos + 1]:
                    continue
                other_used = _all_diffs_except(rows, i)
                candidate_row = row[:]
                candidate_row[pos] = new_val
                new_d = _row_diffs(candidate_row)
                if new_d & other_used:
                    continue
                rows[i] = candidate_row
                found = True
                break
            if found:
                break
        if not found:
            break
    return rows


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

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
    for attempt in range(50):
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
        # tighten
        if _time_left() > 5.0:
            tightened = _iterative_tightening(rows, n_rows, k, rng, max_rounds=20)
            tc = _scope(tightened)
            if tc < best_scope:
                best_scope = tc
                best_rows = tightened

    # Simulated annealing phase
    if _time_left() > 10.0:
        sa_rows = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=400, initial_temp=60.0, cooling=0.97)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
        # Final tightening after SA
        if _time_left() > 5.0:
            tightened = _iterative_tightening(best_rows, n_rows, k, rng, max_rounds=30)
            tc = _scope(tightened)
            if tc < best_scope:
                best_scope = tc
                best_rows = tightened

    # Final aggressive local search
    if _time_left() > 3.0:
        ls = _local_search(best_rows, n_rows, k, rng, max_iters=600)
        if _scope(ls) < best_scope:
            best_rows = ls
            best_scope = _scope(ls)
        tightened = _iterative_tightening(best_rows, n_rows, k, rng, max_rounds=40)
        if _scope(tightened) < best_scope:
            best_rows = tightened
            best_scope = _scope(tightened)

    if not _validate(best_rows, n_rows, k):
        # safe fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)

    return best_rows
