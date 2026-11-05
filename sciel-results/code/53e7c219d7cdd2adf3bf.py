import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140.0  # 10s safety margin under 150s

    def now():
        return time.time()

    def get_scope(rows):
        return max(max(r) for r in rows)

    def row_diffs(row):
        out = []
        n = len(row)
        for i in range(n):
            a = row[i]
            for j in range(i + 1, n):
                out.append(row[j] - a)
        return out

    def full_diffs(rows):
        out = []
        for r in rows:
            out.extend(row_diffs(r))
        return out

    def is_valid(rows):
        d = full_diffs(rows)
        return len(d) == len(set(d))

    def greedy_construct(M):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = [x - a for a in row]
                if any(v in used for v in nd):
                    continue
                row.append(x)
                used.update(nd)
                if x > M:
                    return None
            rows.append(row)
        if get_scope(rows) <= M:
            return rows
        return None

    # ---- CSP: build one row with forward checking + MRV + cost-ordered values ----
    def build_row(used, M, budget):
        row = [0]
        # dom[t] = sorted list of candidate values for position t (t=1..k)
        doms = []
        for t in range(1, k + 1):
            lo = t  # strictly increasing from 0
            hi = M
            doms.append(list(range(lo, hi + 1)))

        # forward check: prune each domain against existing row values
        def prune(dom, fixed):
            res = []
            for x in dom:
                ok = True
                for a in fixed:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    res.append(x)
            return res

        # cost heuristic for a candidate x at position t given current row
        def cost(x, row, M):
            # fragmentation: count of used values in (minrow, x) that are far from x
            c = 0.0
            for a in row:
                d = x - a
                # count used values strictly between a and x
                # (proxy for carving contiguous gaps)
                c += 0.0
            # penalty for large x (prefer compact)
            c += 0.02 * x
            return c

        def backtrack(row, doms, idx, used):
            if idx == k:
                return row[:]
            if now() > budget:
                return None

            # MRV: choose position with smallest domain (from idx onward)
            best_i = None
            best_len = None
            for i in range(idx, k):
                L = len(doms[i])
                if L == 0:
                    return None
                if best_len is None or L < best_len:
                    best_len = L
                    best_i = i
                    if L == 1:
                        break

            # fix positions idx..best_i-1 in order (they must be filled too)
            # To keep it simple, we fill positions sequentially but pick candidate
            # ordering by cost. We always commit to position idx.
            i = idx
            dom = doms[i]
            # order candidates by cost ascending (prefer small / compact)
            fixed = row
            ordered = sorted(dom, key=lambda x: 0.02 * x)

            for x in ordered:
                # conflict check against used (already pruned, but re-verify)
                ok = True
                for a in fixed:
                    if (x - a) in used:
                        ok = False
                        break
                if not ok:
                    continue
                new_used = used | set(x - a for a in fixed)

                # forward-check remaining domains
                new_doms = [None] * k
                new_doms[i] = None  # consumed
                feasible = True
                for j in range(i + 1, k):
                    new_doms[j] = prune(doms[j], fixed + [x])
                    if not new_doms[j]:
                        feasible = False
                        break
                if not feasible:
                    continue

                row.append(x)
                res = backtrack(row, new_doms, i + 1, new_used)
                if res is not None:
                    return res
                row.pop()
            return None

        return backtrack(row, doms, 0, used)

    def build_all(M, budget):
        used = set()
        rows = []
        for _ in range(n_rows):
            if now() > budget:
                return None
            r = build_row(used, M, budget)
            if r is None:
                return None
            rows.append(r)
            for d in row_diffs(r):
                used.add(d)
        return rows

    # ---- local search: single-value mutations that reduce scope ----
    def local_search(rows, M, budget, iters=4000):
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        used = set(full_diffs(best))

        def try_move(ri, pi, new_val):
            # returns new used set if valid, else None
            old = best[ri][pi]
            row = best[ri]
            if new_val <= row[pi - 1] or (pi < k and new_val >= row[pi + 1]):
                return None
            if new_val == old:
                return None
            # diffs removed by old
            removed = []
            for j in range(len(row)):
                if j == pi:
                    continue
                d = abs(row[pi] - row[j])
                removed.append(d)
            # diffs added by new
            added = []
            for j in range(len(row)):
                if j == pi:
                    continue
                d = abs(new_val - row[j])
                added.append(d)
            # check added not colliding with (used - removed)
            tmp = set(used)
            for d in removed:
                tmp.discard(d)
            for d in added:
                if d in tmp:
                    return None
            return (added, removed)

        for _ in range(iters):
            if now() > budget:
                break
            ri = random.randint(0, len(best) - 1)
            pi = random.randint(1, k)
            row = best[ri]
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else M
            if hi < lo:
                continue
            new_val = random.randint(lo, hi)
            res = try_move(ri, pi, new_val)
            if res is None:
                continue
            added, removed = res
            # apply
            best[ri][pi] = new_val
            for d in removed:
                used.discard(d)
            for d in added:
                used.add(d)
            new_scope = get_scope(best)
            if new_scope > best_scope:
                # revert
                best[ri][pi] = row[pi]
                for d in added:
                    used.discard(d)
                for d in removed:
                    used.add(d)
            else:
                best_scope = new_scope
        return best, best_scope

    # ---------- main ----------
    # initial greedy
    best = greedy_construct(100000)
    if best is None:
        # fallback: pure greedy without M cap
        used = set()
        best = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = [x - a for a in row]
                if any(v in used for v in nd):
                    continue
                row.append(x)
                used.update(nd)
            best.append(row)
    best_scope = get_scope(best)

    # try to tighten via CSP from current scope downward
    target = best_scope
    while now() < deadline - 5:
        t = target - 1
        # counting lower bound
        total = n_rows * k * (k + 1) // 2
        if t < total:
            break
        budget = deadline - 8
        res = build_all(t, budget)
        if res is not None and get_scope(res) <= t:
            best = res
            best_scope = t
            target = t
        else:
            # can't go one lower right now; stop descending
            break

    # polish with local search
    if now() < deadline - 5:
        polished, ps = local_search(best, best_scope, deadline - 6)
        if ps < best_scope:
            best = polished
            best_scope = ps

    return best
