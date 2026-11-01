"""Difference triangle set (4,7): minimize the largest entry m(4,7)."""
import time
import random

def _diffs(row):
    d = 0
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d |= (1 << (row[j] - row[i]))
    return d

def _rebuild(row, other_used, k, rng):
    target = row[-1]
    for t in range(1, target):
        for order in (0, 1):
            newrow = [0]
            newused = 0
            ok = True
            for _ in range(k):
                found = False
                if order == 0:
                    cand_range = range(1, t)
                else:
                    cand = list(range(1, t))
                    rng.shuffle(cand)
                    cand_range = cand
                for c in cand_range:
                    new = 0
                    for a in newrow:
                        new |= (1 << (c - a))
                    if new & newused:
                        continue
                    new2 = 0
                    for a in (newrow + [c]):
                        new2 |= (1 << (t - a))
                    if new2 & newused:
                        continue
                    newrow.append(c)
                    newused |= new
                    found = True
                    break
                if not found:
                    ok = False
                    break
            if ok:
                new2 = 0
                for a in newrow:
                    new2 |= (1 << (t - a))
                if new2 & newused:
                    ok = False
                else:
                    newrow.append(t)
                    newused |= new2
            if ok and len(newrow) == k + 1 and newrow[-1] == t:
                return newrow, newused
    return None, None

def construct(n_rows, k):
    start = time.time()
    deadline = start + 118.0
    rng = random.Random(12345)

    def scope_of(rows):
        return max(max(r) for r in rows)

    def greedy_init(seed):
        r = random.Random(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = 0
                for a in row:
                    new |= (1 << (x - a))
                if new & used:
                    continue
                if r.random() < 0.05 and x < 200:
                    continue
                row.append(x)
                used |= new
            rows.append(row)
        return rows, used

    def local_search(rows, used, budget):
        cur_rows = [list(r) for r in rows]
        cur_used = used
        cur_scope = scope_of(cur_rows)
        improved = True
        iters = 0
        while improved and time.time() < budget:
            improved = False
            iters += 1
            for ri in range(n_rows):
                row = cur_rows[ri]
                for ei in range(len(row) - 1, 0, -1):
                    old = row[ei]
                    for newv in range(row[ei - 1] + 1, old):
                        old_diffs = 0
                        for a in row:
                            if a != old:
                                old_diffs |= (1 << (abs(old - a)))
                        new_diffs = 0
                        for a in row:
                            if a != old:
                                new_diffs |= (1 << (abs(newv - a)))
                        if new_diffs & (cur_used & ~old_diffs):
                            continue
                        row[ei] = newv
                        cur_used = (cur_used & ~old_diffs) | new_diffs
                        improved = True
                        break
                    row[ei] = old
            for ri in range(n_rows):
                if time.time() >= budget:
                    break
                row = cur_rows[ri]
                old_row_diffs = _diffs(row)
                other_used = cur_used & ~old_row_diffs
                newrow, newrow_used = _rebuild(row, other_used, k, rng)
                if newrow is not None and newrow[-1] < row[-1]:
                    cur_rows[ri] = newrow
                    cur_used = other_used | newrow_used
                    improved = True
            if time.time() < budget - 2:
                ri = rng.randrange(n_rows)
                old_row_diffs = _diffs(cur_rows[ri])
                other_used = cur_used & ~old_row_diffs
                newrow, newrow_used = _rebuild(cur_rows[ri], other_used, k, rng)
                if newrow is not None:
                    pert = [list(r) for r in cur_rows]
                    pert[ri] = newrow
                    new_used = other_used | newrow_used
                    s = scope_of(pert)
                    if s < cur_scope:
                        cur_rows, cur_used, cur_scope = pert, new_used, s
                        improved = True
            s = scope_of(cur_rows)
            if s < cur_scope:
                cur_scope = s
                improved = True
            if not improved and iters > 3:
                break
        return cur_rows, cur_used, cur_scope

    def polish(rows, used, budget):
        cur_rows = [list(r) for r in rows]
        cur_used = used
        cur_scope = scope_of(cur_rows)
        changed = True
        while changed and time.time() < budget:
            changed = False
            for ri in range(n_rows):
                row = cur_rows[ri]
                for ei in range(len(row) - 1, 0, -1):
                    old = row[ei]
                    for newv in range(row[ei - 1] + 1, old):
                        old_diffs = 0
                        for a in row:
                            if a != old:
                                old_diffs |= (1 << (abs(old - a)))
                        new_diffs = 0
                        for a in row:
                            if a != old:
                                new_diffs |= (1 << (abs(newv - a)))
                        if new_diffs & (cur_used & ~old_diffs):
                            continue
                        row[ei] = newv
                        cur_used = (cur_used & ~old_diffs) | new_diffs
                        cur_scope = min(cur_scope, scope_of(cur_rows))
                        changed = True
                        break
                    row[ei] = old
        return cur_rows, cur_used, cur_scope

    best_rows, best_used = greedy_init(0)
    best_scope = scope_of(best_rows)

    for seed in range(1, 20):
        if time.time() > deadline - 40:
            break
        rows, used = greedy_init(seed)
        s = scope_of(rows)
        if s < best_scope:
            best_rows, best_used, best_scope = [list(r) for r in rows], used, s
        lr, lu, ls = local_search(rows, used, deadline - 30)
        if ls < best_scope:
            best_rows, best_used, best_scope = lr, lu, ls

    lr, lu, ls = local_search(best_rows, best_used, deadline - 20)
    if ls < best_scope:
        best_rows, best_used, best_scope = lr, lu, ls

    pr, pu, ps = polish(best_rows, best_used, deadline - 2)
    if ps < best_scope:
        best_rows, best_used, best_scope = pr, pu, ps

    return best_rows
