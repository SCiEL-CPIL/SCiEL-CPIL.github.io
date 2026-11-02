"""Stronger search for the difference-triangle set (7,5): minimize the largest entry."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _diffs(row):
    """All positive differences a_j - a_i for i < j in a strictly increasing row."""
    n = len(row)
    out = []
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ai)
    return out


def _row_endpoint(row, used, k):
    """Smallest endpoint x such that row+[x] has no diff in `used` and is strictly increasing.
    Returns None if no such x exists below a generous cap."""
    n = k + 1
    last = row[-1]
    x = last + 1
    # cap generously; if we can't find one, we give up (None)
    cap = 10000
    while x <= cap:
        ok = True
        for a in row:
            d = x - a
            if d in used:
                ok = False
                break
        if ok:
            return x
        x += 1
    return None


def _init_rows(n_rows, k, rng):
    """Build rows greedily by smallest feasible endpoint, in a randomized order."""
    used = set()
    rows = [None] * n_rows
    order = list(range(n_rows))
    rng.shuffle(order)
    for idx in order:
        row = [0]
        for _ in range(k):
            x = _row_endpoint(row, used, k)
            if x is None:
                # fallback: linear scan from last+1
                last = row[-1]
                x = last + 1
                while True:
                    if all((x - a) not in used for a in row):
                        break
                    x += 1
            row.append(x)
            used.update(_diffs(row))
        rows[idx] = row
    return rows, used


def _scope(rows):
    return max(r[-1] for r in rows)


def _used_set(rows):
    used = set()
    for r in rows:
        used.update(_diffs(r))
    return used


def _reoptimize_row(row, used, k, cap):
    """Try to rebuild `row` so that its endpoint is as small as possible while keeping
    its diffs disjoint from `used` (the diffs of all OTHER rows). Returns (new_row, diffs)
    or None if it cannot get strictly better than the current endpoint."""
    cur_end = row[-1]
    # greedy build with smallest feasible endpoint, starting from 0
    best = None
    best_end = None
    # try a few randomized tie-breaks to escape local minima
    for _ in range(3):
        new = [0]
        ok = True
        for _ in range(k):
            last = new[-1]
            x = last + 1
            found = None
            while x <= cap:
                if all((x - a) not in used for a in new):
                    found = x
                    break
                x += 1
            if found is None:
                ok = False
                break
            new.append(found)
        if ok:
            end = new[-1]
            if best_end is None or end < best_end:
                best_end = end
                best = new
    if best is None or best_end >= cur_end:
        return None
    return best, set(_diffs(best))


def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 145.0   # safety margin under the 150s kill
    rng = random.Random(12345)

    rows, used = _init_rows(n_rows, k, rng)
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)

    def now():
        return time.time()

    # ---- Simulated annealing ----
    # We track `used` = set of all diffs across rows (for O(1) membership).
    # Move types:
    #   1) shift one element of one row (fast delta)
    #   2) re-optimize the max row (attacks scope directly)
    #   3) swap two rows
    T0 = 10.0
    T_end = 0.05
    total_steps = 250000

    # Precompute row diff lists for fast removal
    row_diffs = [set(_diffs(r)) for r in rows]

    step = 0
    while step < total_steps:
        if now() > deadline:
            break
        T = T0 * (T_end / T0) ** (step / total_steps)

        r = rng.randrange(n_rows)
        move = rng.random()

        if move < 0.55:
            # ---- single element shift ----
            row = rows[r]
            # pick an interior or endpoint index to shift (not index 0)
            pos = rng.randrange(1, k + 1)
            old = row[pos]
            delta_move = rng.randint(-6, 6)
            new = old + delta_move
            if new <= row[pos - 1]:
                step += 1
                continue
            if pos < k and new >= row[pos + 1]:
                step += 1
                continue
            # compute diff changes: removing old diffs involving pos, adding new
            # old diffs involving pos:
            removed = []
            added = []
            ok = True
            for j in range(k + 1):
                if j == pos:
                    continue
                d_old = abs(old - row[j])
                d_new = abs(new - row[j])
                if d_old == d_new:
                    continue
                removed.append(d_old)
                added.append(d_new)
            # check added not in used (excluding the row's own removed set)
            local_removed = set(removed)
            for d in added:
                if d in used and d not in local_removed:
                    ok = False
                    break
            if not ok:
                step += 1
                continue
            # evaluate scope change
            old_scope = _scope(rows)
            # apply
            row[pos] = new
            # update used
            for d in removed:
                used.discard(d)
            for d in added:
                used.add(d)
            new_scope = _scope(rows)
            dscope = new_scope - old_scope
            if dscope <= 0 or rng.random() < math.exp(-dscope / max(T, 1e-9)):
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [rr[:] for rr in rows]
            else:
                # revert
                row[pos] = old
                for d in added:
                    used.discard(d)
                for d in removed:
                    used.add(d)
            row_diffs[r] = set(_diffs(row))

        elif move < 0.80:
            # ---- re-optimize the max row ----
            # find the row with the largest endpoint
            max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            target = rows[max_idx][-1]
            if target < best_scope:
                # nothing to gain; skip
                step += 1
                continue
            # used excluding this row's diffs
            other_used = set(used)
            rd = set(_diffs(rows[max_idx]))
            other_used -= rd
            new_row, new_d = _reoptimize_row(rows[max_idx], other_used, k, cap=target)
            if new_row is None:
                step += 1
                continue
            # apply
            for d in rd:
                used.discard(d)
            for d in new_d:
                used.add(d)
            rows[max_idx] = new_row
            row_diffs[max_idx] = new_d
            ns = _scope(rows)
            if ns < best_scope:
                best_scope = ns
                best_rows = [rr[:] for rr in rows]
        else:
            # ---- swap two rows ----
            a = rng.randrange(n_rows)
            b = rng.randrange(n_rows)
            if a == b:
                step += 1
                continue
            rows[a], rows[b] = rows[b], rows[a]
            row_diffs[a], row_diffs[b] = row_diffs[b], row_diffs[a]
            # scope unchanged; always accept (neutral)
            ns = _scope(rows)
            if ns < best_scope:
                best_scope = ns
                best_rows = [rr[:] for rr in rows]

        step += 1

    # ---- Final tightening loop: keep lowering scope via re-optimization ----
    for _ in range(40):
        if now() > deadline:
            break
        improved = False
        for i in range(n_rows):
            if now() > deadline:
                break
            target = rows[i][-1]
            if target <= best_scope:
                continue
            other_used = set(used)
            rd = set(_diffs(rows[i]))
            other_used -= rd
            new_row, new_d = _reoptimize_row(rows[i], other_used, k, cap=target)
            if new_row is None:
                continue
            for d in rd:
                used.discard(d)
            for d in new_d:
                used.add(d)
            rows[i] = new_row
            row_diffs[i] = new_d
            improved = True
            ns = _scope(rows)
            if ns < best_scope:
                best_scope = ns
                best_rows = [rr[:] for rr in rows]
        if not improved:
            break

    # Prefer the best valid result found
    result = best_rows
    # Validate result (defensive): all positive diffs distinct across rows
    seen = set()
    ok = True
    for r in result:
        if r[0] != 0:
            ok = False
            break
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in seen:
                    ok = False
                seen.add(d)
    if not ok:
        # fall back to current rows if they are valid
        seen2 = set()
        ok2 = True
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen2:
                        ok2 = False
                    seen2.add(d)
        if ok2:
            result = rows
        else:
            # last resort: rebuild fresh greedy
            result, _ = _init_rows(n_rows, k, rng)

    return result

# EVOLVE-BLOCK-END
