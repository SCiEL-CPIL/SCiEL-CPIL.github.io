import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 147.0  # safety margin under the 150 s kill

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
        for _ in range(5000):
            if time.time() > deadline:
                break
            classes, shadows = shadows_of(col, conflict)
            merged = False
            ckeys = list(classes.keys())
            # Try merges, preferring those that result in smaller combined shadow
            best_merge = None
            best_merge_shadow_size = float('inf')
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
                    # Combined shadow size after merge
                    comb_shadow = shadows[a] | shadows[b]
                    sz = len(comb_shadow)
                    if sz < best_merge_shadow_size:
                        best_merge_shadow_size = sz
                        best_merge = (a, b)
            if best_merge:
                a, b = best_merge
                for e in classes[b]:
                    col[e] = a
                del classes[b]
                merged = True
            if merged:
                no_merge_streak = 0
                no_move_streak = 0
                k = num_colours(col)
                if k < best_k:
                    best_k = k
                    best = col[:]
                continue
            # no merge: try single-edge moves first
            no_merge_streak += 1
            if no_merge_streak > 120:
                break
            classes, shadows = shadows_of(col, conflict)
            ckeys = list(classes.keys())
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
                no_move_streak = 0
                continue
            # Try 2-swap: exchange edges between two colour classes
            no_move_streak += 1
            if no_move_streak > 30:
                break
            classes, shadows = shadows_of(col, conflict)
            ckeys = list(classes.keys())
            swapped = False
            for a in ckeys:
                if a not in classes or len(classes[a]) < 1:
                    continue
                for b in ckeys:
                    if a >= b or b not in classes or len(classes[b]) < 1:
                        continue
                    ea_list = list(classes[a])
                    eb_list = list(classes[b])
                    for ea in ea_list:
                        for eb in eb_list:
                            # Check if swapping ea to b and eb to a is valid
                            ok = True
                            for x in classes[a]:
                                if x == ea:
                                    continue
                                if eb in conflict[x]:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            for x in classes[b]:
                                if x == eb:
                                    continue
                                if ea in conflict[x]:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            # Perform swap
                            col[ea] = b
                            col[eb] = a
                            classes[a].discard(ea)
                            classes[a].add(eb)
                            classes[b].discard(eb)
                            classes[b].add(ea)
                            swapped = True
                            break
                        if swapped:
                            break
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                break
        # Final polish: try to eliminate colour classes one by one
        for _ in range(50):
            if time.time() > deadline:
                break
            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys(), key=lambda c: len(classes[c]))
            eliminated = False
            for c in ckeys:
                if c not in classes:
                    continue
                # Try to reassign all edges of colour c to other colours
                edges_c = list(classes[c])
                other_classes = [d for d in classes if d != c]
                if not other_classes:
                    break
                assignment = {}
                feasible = True
                for e in edges_c:
                    placed = False
                    for d in other_classes:
                        # Check if e can go to colour d
                        can_place = True
                        for x in classes[d]:
                            if e in conflict[x]:
                                can_place = False
                                break
                        if can_place:
                            assignment[e] = d
                            placed = True
                            break
                    if not placed:
                        feasible = False
                        break
                if feasible:
                    for e, d in assignment.items():
                        col[e] = d
                    eliminated = True
                    break
            if not eliminated:
                break
        if is_valid(col, conflict):
            k = num_colours(col)
            if k < best_k:
                best_k = k
                best = col[:]
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
            sorted(range(m), key=lambda i: (-cd[i], i)),
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

    def dsatur_color(conflict, m, deadline, rng):
        """
        DSATUR (Dynamic Saturation) heuristic for vertex coloring of the conflict graph.
        This is structurally different from first-fit greedy: it dynamically selects
        the next vertex based on saturation (number of distinct colors in its neighborhood)
        rather than using a fixed or random order.
        """
        col = [-1] * m
        # sat[i] = set of colors used by neighbors of i
        sat = [set() for _ in range(m)]
        remaining = set(range(m))
        
        for _ in range(m):
            if time.time() > deadline:
                break
            # Select vertex with highest saturation, then highest degree in remaining
            best_v = -1
            best_key = None
            for v in remaining:
                sat_size = len(sat[v])
                # Count neighbors in remaining
                deg_rem = sum(1 for u in conflict[v] if u in remaining)
                key = (sat_size, deg_rem, rng.random())
                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v
            
            if best_v == -1:
                break
            
            # Assign smallest available color
            used = set(sat[best_v])
            c = 0
            while c in used:
                c += 1
            col[best_v] = c
            remaining.discard(best_v)
            # Update saturation of neighbors
            for u in conflict[best_v]:
                if u in remaining:
                    sat[u].add(c)
        
        # Handle any uncolored vertices (shouldn't happen if we run to completion)
        for v in remaining:
            used = set(sat[v])
            c = 0
            while c in used:
                c += 1
            col[v] = c
            for u in conflict[v]:
                if u in remaining:
                    sat[u].add(c)
        
        return col

    def solve_instance_dsatur(edges, deadline):
        m = len(edges)
        n, adj, inc, conflict = build(edges)
        
        # Generate multiple DSATUR solutions with different random tie-breaking
        cand = []
        seen = set()
        
        def add_cand(col):
            sig = tuple(col)
            if sig in seen:
                return
            seen.add(sig)
            cand.append((num_colours(col), col[:]))
        
        # Run DSATUR with different random seeds for tie-breaking
        n_runs = 20
        time_per_run = (deadline - time.time()) / max(1, n_runs + 5)
        
        for run in range(n_runs):
            if time.time() > deadline - 1.0:
                break
            rng = Random(1000 + run)
            run_dl = min(deadline - 0.5, time.time() + time_per_run * 2)
            col = dsatur_color(conflict, m, run_dl, rng)
            add_cand(col)
        
        # Also include a few greedy solutions for diversity
        cd = [len(conflict[i]) for i in range(m)]
        base_orders = [
            list(range(m)),
            sorted(range(m), key=lambda i: -cd[i]),
            sorted(range(m), key=lambda i: cd[i]),
        ]
        for o in base_orders:
            add_cand(greedy(o, conflict, m))
        
        if not cand:
            col = greedy(list(range(m)), conflict, m)
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            return [remap[c] for c in col]
        
        cand.sort(key=lambda x: x[0])
        top = []
        tsig = set()
        for k, col in cand:
            s = tuple(col)
            if s in tsig:
                continue
            tsig.add(s)
            top.append((k, col))
            if len(top) >= 4:
                break
        
        best = None
        best_k = 10 ** 9
        n_ls_runs = len(top)
        for k, col in top:
            if time.time() > deadline - 1.0:
                break
            slice_dl = min(deadline - 0.5, time.time() + max(0.5, (deadline - time.time()) / max(1, n_ls_runs)))
            res = local_search(col, conflict, m, slice_dl)
            rk = num_colours(res)
            if rk < best_k:
                best_k = rk
                best = res
            n_ls_runs -= 1
        
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
        inst_dl = min(DEADLINE - 3.0 - 2.0 * left_after, time.time() + 45.0)
        if inst_dl < time.time() + 5.0:
            inst_dl = min(DEADLINE - 1.0, time.time() + 5.0)
        try:
            # Use DSATUR-based approach
            out.append(solve_instance_dsatur(edges, inst_dl))
        except Exception:
            # fast, always-valid fallback
            n, adj, inc, conflict = build(edges)
            col = greedy(list(range(m)), conflict, m)
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            out.append([remap[c] for c in col])
    return out
