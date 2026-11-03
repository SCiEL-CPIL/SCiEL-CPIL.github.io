import random
import time
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    def get_scope(rows):
        return max((max(r) for r in rows), default=0)

    def is_valid_solution(rows):
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if all((x - a) not in used for a in row):
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    # ---- Bitset helpers for difference masks ----
    def diff_mask(row):
        m = 0
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                m |= 1 << (row[j] - ri)
        return m

    def row_max(row):
        return row[-1]

    # ---- Pattern generation (diverse Sidon rows within scope) ----
    def gen_greedy(used, scope, rng):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > scope:
                return None
            if all(((x - a) not in used) for a in row):
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        return row

    def gen_random(used, scope, rng):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > scope:
                return None
            if all(((x - a) not in used) for a in row):
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        # randomize internal values to create diversity
        for _ in range(6):
            idx = rng.randint(1, len(row) - 1)
            lo = row[idx - 1] + 1
            hi = row[idx + 1] - 1 if idx < len(row) - 1 else scope
            if lo > hi:
                continue
            newv = rng.randint(lo, hi)
            if newv == row[idx]:
                continue
            # check no new diff collision with this row's own diffs
            newdiffs = set()
            for i in range(idx):
                newdiffs.add(newv - row[i])
            for j in range(idx + 1, len(row)):
                newdiffs.add(row[j] - newv)
            # diffs within row must stay distinct
            if len(newdiffs) != idx + (len(row) - 1 - idx):
                continue
            # must not collide with other rows' diffs
            if newdiffs & used:
                continue
            # remove old diffs of this element
            for i in range(idx):
                used.discard(row[idx] - row[i])
            for j in range(idx + 1, len(row)):
                used.discard(row[j] - row[idx])
            row[idx] = newv
            used |= newdiffs
        return row

    def gen_backtrack(used, scope, rng, depth=0):
        # build a Sidon row with backtracking, randomizing choices
        row = [0]
        row_used = set()

        def rec():
            if len(row) == k + 1:
                return True
            start = row[-1] + 1
            cands = []
            for x in range(start, scope + 1):
                ok = True
                for a in row:
                    if (x - a) in used or (x - a) in row_used:
                        ok = False
                        break
                if ok:
                    cands.append(x)
            if not cands:
                return False
            rng.shuffle(cands)
            for x in cands:
                nd = {x - a for a in row}
                row.append(x)
                row_used |= nd
                if rec():
                    return True
                row.pop()
                for d in nd:
                    row_used.discard(d)
            return False

        if rec():
            return row
        return None

    def generate_patterns(scope, budget, rng):
        """Return list of (mask, row) for diverse Sidon rows within scope."""
        patterns = {}
        target = 15000
        t0 = time.time()
        gens = 0
        while len(patterns) < target and time.time() < budget and gens < 60000:
            gens += 1
            used = set()
            r = gen_greedy(used, scope, rng)
            if r is None:
                continue
            m = diff_mask(r)
            key = (m, row_max(r))
            if key in patterns:
                continue
            patterns[key] = r
            # create a diverse variant
            used2 = set()
            used2 |= {d for d in range(1, scope + 1) if (m >> d) & 1}
            r2 = gen_random(used2, scope, rng)
            if r2 is not None:
                m2 = diff_mask(r2)
                k2 = (m2, row_max(r2))
                if k2 not in patterns:
                    patterns[k2] = r2
        # add some backtracked ones
        for _ in range(400):
            if len(patterns) >= target or not time_ok():
                break
            used = set()
            r = gen_backtrack(used, scope, rng)
            if r is None:
                continue
            m = diff_mask(r)
            key = (m, row_max(r))
            if key not in patterns:
                patterns[key] = r
        return [(m, r) for (m, r) in patterns.items()]

    # ---- Maximum Independent Set (branch & bound, Tomita coloring) ----
    def max_independent_set(adj, n, need, deadline):
        """adj: list of int bitmasks; return list of chosen vertices or None."""
        best = None
        best_size = 0

        def greedy_color(subset, order):
            # color the vertices in `order` (list of vertex ids) greedily
            color_of = {}
            color_sets = []  # list of lists
            for v in order:
                placed = False
                for ci in range(len(color_sets)):
                    ok = True
                    for u in color_sets[ci]:
                        if (adj[v] >> u) & 1:
                            ok = False
                            break
                    if ok:
                        color_sets[ci].append(v)
                        color_of[v] = ci
                        placed = True
                        break
                if not placed:
                    color_sets.append([v])
                    color_of[v] = len(color_sets) - 1
            return color_of, len(color_sets)

        def expand(P, C, chosen):
            nonlocal best, best_size
            if not P:
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best = chosen[:]
                return
            if len(chosen) + C <= best_size:
                return
            if best_size >= need and len(chosen) + C < need:
                return
            # choose a vertex
            v = P[0]
            P = P[1:]
            # branch 1: include v
            chosen.append(v)
            Pn = [u for u in P if (adj[v] >> u) & 1]
            # recompute color bound for Pn
            if Pn:
                _, cb = greedy_color(Pn, Pn)
            else:
                cb = 0
            expand(Pn, cb, chosen)
            chosen.pop()
            # branch 2: exclude v
            expand(P, C - 1, chosen)

        # initial coloring
        order = list(range(n))
        random.Random(0).shuffle(order)
        _, cb = greedy_color(order, order)
        expand(order, cb, [])
        return best

    def find_packing(scope, rng):
        """Try to find n_rows compatible rows within scope. Return list of rows or None."""
        budget = min(time.time() + 25.0, deadline - 3.0)
        if not time_ok():
            return None
        pats = generate_patterns(scope, budget, rng)
        if len(pats) < n_rows:
            return None
        # sort by max value ascending (prefer compact)
        pats.sort(key=lambda pr: (pr[1][-1], pr[0]))
        n = len(pats)
        masks = [p[0] for p in pats]
        rows = [p[1] for p in pats]
        # build adjacency (compatible = disjoint)
        adj = [0] * n
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                if mi & masks[j] == 0:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        # If there are too many, subsample to keep MIS tractable
        max_nodes = 4000
        if n > max_nodes:
            # keep the most compact ones
            pats = pats[:max_nodes]
            n = max_nodes
            masks = masks[:n]
            rows = rows[:n]
            adj = [0] * n
            for i in range(n):
                mi = masks[i]
                for j in range(i + 1, n):
                    if mi & masks[j] == 0:
                        adj[i] |= 1 << j
                        adj[j] |= 1 << i
        chosen = max_independent_set(adj, n, n_rows, deadline)
        if chosen is None or len(chosen) < n_rows:
            return None
        return [rows[i][:] for i in chosen[:n_rows]]

    # ---- Bitset local compaction to shave scope ----
    def compact(rows, deadline):
        rows = [r[:] for r in rows]
        masks = [diff_mask(r) for r in rows]
        used = 0
        for m in masks:
            used |= m
        scope = get_scope(rows)
        rng = random.Random(123)
        while time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                row = rows[ri]
                for idx in range(len(row) - 1, 0, -1):
                    lo = row[idx - 1] + 1
                    hi = row[idx] - 1
                    if lo > hi:
                        continue
                    # try to move row[idx] down
                    m = masks[ri]
                    old = row[idx]
                    # diffs contributed by element idx
                    old_diffs = 0
                    for i in range(idx):
                        old_diffs |= 1 << (old - row[i])
                    for j in range(idx + 1, len(row)):
                        old_diffs |= 1 << (row[j] - old)
                    for newv in range(hi, lo - 1, -1):
                        if newv == old:
                            continue
                        new_diffs = 0
                        for i in range(idx):
                            new_diffs |= 1 << (newv - row[i])
                        for j in range(idx + 1, len(row)):
                            new_diffs |= 1 << (row[j] - newv)
                        # internal distinctness: new_diffs must not overlap each other (they are distinct by construction of row)
                        # check global disjointness
                        rest = (used & ~m)
                        if new_diffs & rest:
                            continue
                        # apply
                        used = (used & ~old_diffs) | new_diffs
                        row[idx] = newv
                        masks[ri] = diff_mask(row)
                        if newv < old:
                            improved = True
                            scope = min(scope, newv)
                        break
            if not improved:
                break
        return rows

    # ---- Main ----
    rng = random.Random(42)
    current = greedy_init()
    best_rows = [r[:] for r in current]
    best_scope = get_scope(best_rows)

    # local compaction of greedy
    if time_ok():
        comp = compact(best_rows, deadline - 5.0)
        if is_valid_solution(comp) and get_scope(comp) < best_scope:
            best_scope = get_scope(comp)
            best_rows = comp

    # binary search for the smallest scope with a full packing
    lo = 112  # counting bound (n_rows * k*(k+1)/2)
    hi = best_scope
    # first ensure hi is feasible (it is, best_rows works)
    found_hi = True
    # try to find a feasible scope a bit below best to tighten
    while lo < hi and time.time() < deadline - 8.0:
        mid = (lo + hi) // 2
        res = find_packing(mid, rng)
        if res is not None and is_valid_solution(res):
            sc = get_scope(res)
            if sc < best_scope:
                best_scope = sc
                best_rows = res
            hi = mid
        else:
            lo = mid + 1

    # final polish on best
    if time_ok():
        comp = compact(best_rows, deadline - 2.0)
        if is_valid_solution(comp) and get_scope(comp) < best_scope:
            best_scope = get_scope(comp)
            best_rows = comp

    if not is_valid_solution(best_rows):
        best_rows = greedy_init()

    return best_rows
