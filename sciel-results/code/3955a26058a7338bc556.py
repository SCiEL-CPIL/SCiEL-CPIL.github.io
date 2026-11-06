import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build(edges):
        m = len(edges)
        if m == 0:
            return 0, [], [], []
        n = max(max(u, v) for u, v in edges) + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        inc = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            inc[u].add(i)
            inc[v].add(i)
        # conflict[e] = edges f != e with an endpoint in adj[u] | adj[v]  (e=(u,v))
        conflict = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            forb = adj[u] | adj[v]
            s = set()
            for w in forb:
                s |= inc[w]
            s.discard(i)
            conflict[i] = s
        return n, adj, inc, conflict

    def greedy(order, conflict, m):
        col = [-1] * m
        for i in order:
            used = set()
            for j in conflict[i]:
                cj = col[j]
                if cj >= 0:
                    used.add(cj)
            c = 0
            while c in used:
                c += 1
            col[i] = c
        return col

    def num_colours(col):
        return len(set(col))

    def is_valid(col, conflict):
        classes = {}
        for i, c in enumerate(col):
            classes.setdefault(c, set()).add(i)
        for es in classes.values():
            for i in es:
                for j in es:
                    if i != j and j in conflict[i]:
                        return False
        return True

    def shadows_of(col, conflict):
        classes = {}
        for i, c in enumerate(col):
            classes.setdefault(c, set()).add(i)
        shadows = {}
        for c, es in classes.items():
            s = set()
            for e in es:
                s |= conflict[e]
            shadows[c] = s
        return classes, shadows

    def local_search(col, conflict, m, deadline):
        col = col[:]
        best = col[:]
        best_k = num_colours(col)
        no_merge_streak = 0
        no_move_streak = 0
        for it in range(8000):
            if time.time() > deadline:
                break
            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys())
            merged = False
            for a in ckeys:
                if a not in classes:
                    continue
                for b in ckeys:
                    if a >= b or b not in classes:
                        continue
                    if shadows[a] & classes[b]:
                        continue
                    if shadows[b] & classes[a]:
                        continue
                    for e in classes[b]:
                        col[e] = a
                    del classes[b]
                    merged = True
                    break
                if merged:
                    break
            if merged:
                no_merge_streak = 0
                no_move_streak = 0
                k = num_colours(col)
                if k < best_k:
                    best_k = k
                    best = col[:]
                continue
            # no merge: try single-edge move
            no_merge_streak += 1
            no_move_streak += 1
            if no_merge_streak > 120:
                break
            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys())
            moved = False
            for a in ckeys:
                if a not in classes:
                    continue
                for e in list(classes[a]):
                    for d in ckeys:
                        if d == a or d not in classes:
                            continue
                        if e in shadows[d]:
                            continue
                        col[e] = d
                        moved = True
                        break
                    if moved:
                        break
                if moved:
                    break
            if moved:
                continue
            # no single move: try 2-swap
            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys())
            swapped = False
            for a in ckeys:
                if a not in classes:
                    continue
                ca = classes[a]
                for b in ckeys:
                    if a >= b or b not in classes:
                        continue
                    cb = classes[b]
                    for e1 in list(ca):
                        # precompute: edges in cb that conflict with e1
                        bad1 = conflict[e1] & cb
                        if len(bad1) > 0:
                            continue
                        for e2 in list(cb):
                            bad2 = conflict[e2] & ca
                            if len(bad2) > 0:
                                continue
                            col[e1] = b
                            col[e2] = a
                            swapped = True
                            break
                        if swapped:
                            break
                    if swapped:
                        break
                if swapped:
                    break
            if swapped:
                continue
            # no 2-swap: try remove-class-and-reinsert escape
            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys())
            if len(ckeys) >= 3:
                # pick the largest class to remove
                target = max(ckeys, key=lambda c: len(classes[c]))
                removed = list(classes[target])
                for e in removed:
                    col[e] = -1  # uncolored
                # reinsert each edge greedily into remaining classes
                remaining = [c for c in ckeys if c != target]
                for e in removed:
                    placed = False
                    for d in remaining:
                        if d not in classes:
                            continue
                        if e in shadows.get(d, set()):
                            continue
                        col[e] = d
                        placed = True
                        break
                    if not placed:
                        # assign to a new color (use the removed color)
                        col[e] = target
                k = num_colours(col)
                if k < best_k:
                    best_k = k
                    best = col[:]
                no_merge_streak = 0
                no_move_streak = 0
                continue
            break
        if is_valid(best, conflict):
            return best
        return col

    def solve_instance(edges, deadline):
        m = len(edges)
        n, adj, inc, conflict = build(edges)
        rng = Random(777 + m)
        cd = [len(conflict[i]) for i in range(m)]
        base_orders = [
            list(range(m)),
            sorted(range(m), key=lambda i: -cd[i]),
            sorted(range(m), key=lambda i: cd[i]),
            sorted(range(m), key=lambda i: (cd[i], i)),
        ]
        cand = []
        seen = set()

        def add_cand(col):
            sig = tuple(col)
            if sig in seen:
                return
            seen.add(sig)
            cand.append((num_colours(col), col[:]))

        for o in base_orders:
            add_cand(greedy(o, conflict, m))
        # Add degree-based and random-priority orderings
        base_orders.append(sorted(range(m), key=lambda i: (cd[i], i)))
        for o in base_orders[3:]:
            add_cand(greedy(o, conflict, m))
        t_re = time.time() + 3.0
        cnt = 0
        while time.time() < t_re and cnt < 600:
            o = list(range(m))
            rng.shuffle(o)
            add_cand(greedy(o, conflict, m))
            cnt += 1
        cand.sort(key=lambda x: x[0])
        top = []
        tsig = set()
        for k, col in cand:
            s = tuple(col)
            if s in tsig:
                continue
            tsig.add(s)
            top.append((k, col))
            if len(top) >= 6:
                break
        best = None
        best_k = 10 ** 9
        n_runs = len(top)
        for k, col in top:
            if time.time() > deadline - 1.0:
                break
            slice_dl = min(deadline - 0.5, time.time() + max(0.5, (deadline - time.time()) / max(1, n_runs)))
            res = local_search(col, conflict, m, slice_dl)
            rk = num_colours(res)
            if rk < best_k:
                best_k = rk
                best = res
            n_runs -= 1
        if best is None:
            best = top[0][1]
        if not is_valid(best, conflict):
            best = greedy(list(range(m)), conflict, m)
        present = sorted(set(best))
        remap = {old: new for new, old in enumerate(present)}
        return [remap[c] for c in best]

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    for idx, spec in enumerate(specs):
        edges = gen_instance(*spec)
        m = len(edges)
        left_after = len(specs) - 1 - idx
        inst_dl = min(DEADLINE - 3.0 - 2.0 * left_after, time.time() + 50.0)
        if inst_dl < time.time() + 5.0:
            inst_dl = min(DEADLINE - 1.0, time.time() + 5.0)
        try:
            out.append(solve_instance(edges, inst_dl))
        except Exception:
            # fast, always-valid fallback
            n, adj, inc, conflict = build(edges)
            col = greedy(list(range(m)), conflict, m)
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            out.append([remap[c] for c in col])
    return out
