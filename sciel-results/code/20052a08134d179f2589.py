def construct():
    import time
    from random import Random
    from collections import defaultdict

    t0 = time.time()
    DEADLINE = t0 + 115.0

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

    def build_conflict(edges):
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        conf = [set() for _ in range(m)]
        v_to_e = {}
        for i, (u, v) in enumerate(edges):
            v_to_e.setdefault(u, []).append(i)
            v_to_e.setdefault(v, []).append(i)
        for i in range(m):
            u, v = edges[i]
            near = {u, v} | (adj.get(u, set()) | adj.get(v, set()))
            for x in near:
                for j in v_to_e.get(x, []):
                    if j != i and j > i:
                        conf[i].add(j)
                        conf[j].add(i)
        conf_list = [tuple(sorted(c)) for c in conf]
        return conf_list

    def dsatur_coloring(conf, m, deadline, rng=None, variant=0):
        col = [-1] * m
        forbidden = [set() for _ in range(m)]
        remaining = set(range(m))
        deg = [len(c) for c in conf]
        
        while remaining and time.time() < deadline:
            best_edge = None
            best_key = None
            for e in remaining:
                cnt = len(forbidden[e])
                if variant == 0:
                    key = (-cnt, -deg[e], e)
                elif variant == 1:
                    key = (-cnt, deg[e], -e)
                elif variant == 2:
                    key = (-cnt, -deg[e], -e)
                else:
                    key = (-cnt, -deg[e], e)
                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = e
            
            if best_edge is None:
                best_edge = next(iter(remaining))
            
            c = 0
            while c in forbidden[best_edge]:
                c += 1
            
            col[best_edge] = c
            remaining.discard(best_edge)
            
            for j in conf[best_edge]:
                if j in remaining:
                    forbidden[j].add(c)
        
        return col

    def k_of(col):
        if not col:
            return 0
        return max(col) + 1

    def build_classes(col):
        cls = defaultdict(set)
        for i, c in enumerate(col):
            cls[c].add(i)
        return cls

    def reindex(col):
        present = sorted(set(col))
        if not present:
            return col
        remap = {c: i for i, c in enumerate(present)}
        for i in range(len(col)):
            col[i] = remap[col[i]]
        return col

    def is_valid_move(col, e, new_c, conf):
        for j in conf[e]:
            if col[j] == new_c:
                return False
        return True

    def try_merge_all_pairs(cls, col, conf):
        merges = 0
        while True:
            merged = False
            cols = sorted(cls.keys())
            for idx, c1 in enumerate(cols):
                if c1 not in cls:
                    continue
                s1 = cls[c1]
                for c2 in cols[idx + 1:]:
                    if c2 not in cls:
                        continue
                    s2 = cls[c2]
                    ok = True
                    for e in s2:
                        for j in conf[e]:
                            if j in s1:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for e in s2:
                            col[e] = c1
                        s1.update(s2)
                        del cls[c2]
                        merged = True
                        merges += 1
                        break
                if merged:
                    break
            if not merged:
                break
        return merges

    def local_improve(col, conf, m, deadline, rng, max_guard=1200):
        col = list(col)
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < max_guard:
            improved = False
            guard += 1
            cls = build_classes(col)
            
            n_merges = try_merge_all_pairs(cls, col, conf)
            if n_merges > 0:
                improved = True

            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    moved = False
                    for e in range(m):
                        c = col[e]
                        if c == 0:
                            continue
                        for new_c in range(c):
                            if is_valid_move(col, e, new_c, conf):
                                col[e] = new_c
                                moved = True
                                break
                    if moved:
                        improved = True
                    else:
                        for _ in range(5000):
                            e = rng.randrange(m)
                            c = col[e]
                            if c == 0:
                                continue
                            for new_c in range(c):
                                if is_valid_move(col, e, new_c, conf):
                                    col[e] = new_c
                                    moved = True
                                    break
                            if moved:
                                break
                        if moved:
                            improved = True

            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    swapped = False
                    for _ in range(6000):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        if e1 == e2:
                            continue
                        c1, c2 = col[e1], col[e2]
                        if c1 == c2:
                            continue
                        valid = True
                        for j in conf[e1]:
                            if j != e2 and col[j] == c2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if j != e1 and col[j] == c1:
                                    valid = False
                                    break
                        if valid:
                            col[e1], col[e2] = c2, c1
                            swapped = True
                            break
                    if swapped:
                        improved = True

            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 3:
                    swapped = False
                    for _ in range(3000):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        e3 = rng.randrange(m)
                        if e1 == e2 or e2 == e3 or e1 == e3:
                            continue
                        c1, c2, c3 = col[e1], col[e2], col[e3]
                        if len({c1, c2, c3}) < 3:
                            continue
                        valid = True
                        for j in conf[e1]:
                            if j != e2 and j != e3 and col[j] == c2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if j != e1 and j != e3 and col[j] == c3:
                                    valid = False
                                    break
                        if valid:
                            for j in conf[e3]:
                                if j != e1 and j != e2 and col[j] == c1:
                                    valid = False
                                    break
                        if valid:
                            col[e1], col[e2], col[e3] = c2, c3, c1
                            swapped = True
                            break
                    if swapped:
                        improved = True

        return reindex(col)

    def simulated_annealing(col, conf, m, deadline, rng, iterations=200000):
        col = list(col)
        best_col = col[:]
        best_k = k_of(col)
        T = 1.5
        cool = 0.99995
        n_accept = 0
        max_c = best_k + 10
        color_counts = [0] * max_c
        for c in col:
            if c >= max_c:
                while c >= len(color_counts):
                    color_counts.append(0)
                max_c = len(color_counts)
            color_counts[c] += 1
        
        run_max = best_k - 1

        def update_count(old_c, new_c):
            nonlocal run_max, max_c
            color_counts[old_c] -= 1
            if new_c >= len(color_counts):
                while new_c >= len(color_counts):
                    color_counts.append(0)
                max_c = len(color_counts)
            color_counts[new_c] += 1
            if new_c > run_max:
                run_max = new_c

        def current_k():
            nonlocal run_max
            for i in range(run_max, -1, -1):
                if color_counts[i] > 0:
                    run_max = i
                    return i + 1
            return 0

        def try_single_move(e):
            nonlocal col, color_counts, run_max, max_c, best_col, best_k, n_accept, T
            c_old = col[e]
            r = rng.random()
            if r < 0.4:
                new_c = rng.randrange(c_old) if c_old > 0 else 0
            elif r < 0.7:
                new_c = rng.randrange(best_k + 2)
            else:
                new_c = rng.randrange(max_c)
            
            if new_c == c_old:
                return False
            
            valid = True
            for j in conf[e]:
                if col[j] == new_c:
                    valid = False
                    break
            
            if not valid:
                return False
            
            col[e] = new_c
            update_count(c_old, new_c)
            new_k = current_k()
            
            if new_k < best_k:
                best_col = col[:]
                best_k = new_k
                n_accept += 1
                T = max(0.01, T * 0.97)
            else:
                delta = new_k - best_k
                if delta == 0:
                    if rng.random() < 0.35:
                        n_accept += 1
                    else:
                        col[e] = c_old
                        update_count(new_c, c_old)
                else:
                    if rng.random() < (2.718281828 ** (-delta / T)):
                        n_accept += 1
                    else:
                        col[e] = c_old
                        update_count(new_c, c_old)
            T = max(0.01, T * cool)
            return True

        def try_2swap():
            nonlocal col, color_counts, run_max, max_c, best_col, best_k, n_accept, T
            if m < 2:
                return False
            e1 = rng.randrange(m)
            e2 = rng.randrange(m)
            if e1 == e2:
                return False
            c1, c2 = col[e1], col[e2]
            if c1 == c2:
                return False
            valid = True
            for j in conf[e1]:
                if j != e2 and col[j] == c2:
                    valid = False
                    break
            if valid:
                for j in conf[e2]:
                    if j != e1 and col[j] == c1:
                        valid = False
                        break
            if not valid:
                return False
            
            col[e1], col[e2] = c2, c1
            update_count(c1, c2)
            update_count(c2, c1)
            new_k = current_k()
            
            if new_k < best_k:
                best_col = col[:]
                best_k = new_k
                n_accept += 1
                T = max(0.01, T * 0.97)
            else:
                delta = new_k - best_k
                if delta == 0:
                    if rng.random() < 0.35:
                        n_accept += 1
                    else:
                        col[e1], col[e2] = c1, c2
                        update_count(c2, c1)
                        update_count(c1, c2)
                else:
                    if rng.random() < (2.718281828 ** (-delta / T)):
                        n_accept += 1
                    else:
                        col[e1], col[e2] = c1, c2
                        update_count(c2, c1)
                        update_count(c1, c2)
            T = max(0.01, T * cool)
            return True

        def try_3swap():
            nonlocal col, color_counts, run_max, max_c, best_col, best_k, n_accept, T
            if m < 3:
                return False
            e1 = rng.randrange(m)
            e2 = rng.randrange(m)
            e3 = rng.randrange(m)
            if e1 == e2 or e2 == e3 or e1 == e3:
                return False
            c1, c2, c3 = col[e1], col[e2], col[e3]
            if len({c1, c2, c3}) < 3:
                return False
            
            valid = True
            for j in conf[e1]:
                if j != e2 and j != e3 and col[j] == c2:
                    valid = False
                    break
            if valid:
                for j in conf[e2]:
                    if j != e1 and j != e3 and col[j] == c3:
                        valid = False
                        break
            if valid:
                for j in conf[e3]:
                    if j != e1 and j != e2 and col[j] == c1:
                        valid = False
                        break
            if not valid:
                return False
            
            col[e1], col[e2], col[e3] = c2, c3, c1
            update_count(c1, c2)
            update_count(c2, c3)
            update_count(c3, c1)
            new_k = current_k()
            
            if new_k < best_k:
                best_col = col[:]
                best_k = new_k
                n_accept += 1
                T = max(0.01, T * 0.97)
            else:
                delta = new_k - best_k
                if delta == 0:
                    if rng.random() < 0.3:
                        n_accept += 1
                    else:
                        col[e1], col[e2], col[e3] = c1, c2, c3
                        update_count(c2, c1)
                        update_count(c3, c2)
                        update_count(c1, c3)
                else:
                    if rng.random() < (2.718281828 ** (-delta / T)):
                        n_accept += 1
                    else:
                        col[e1], col[e2], col[e3] = c1, c2, c3
                        update_count(c2, c1)
                        update_count(c3, c2)
                        update_count(c1, c3)
            T = max(0.01, T * cool)
            return True

        for it in range(iterations):
            if time.time() > deadline:
                break
            
            r = rng.random()
            if r < 0.55:
                e = rng.randrange(m)
                try_single_move(e)
            elif r < 0.80:
                try_2swap()
            else:
                try_3swap()
        
        return best_col

    def final_polish(col, conf, m, deadline, rng):
        col = list(col)
        for _ in range(20):
            if time.time() > deadline:
                break
            col = local_improve(col, conf, m, deadline, rng, max_guard=3000)
        for _ in range(8):
            if time.time() < deadline:
                cls = build_classes(col)
                n = try_merge_all_pairs(cls, col, conf)
                if n == 0:
                    break
        return reindex(col)

    def solve_one(spec, time_budget):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conf = build_conflict(edges)
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Phase 1: Multiple DSATUR trials with different variants
        for trial in range(20):
            if time.time() > t0 + time_budget:
                break
            rng = Random(seed + 1000 * trial + 42)
            variant = trial % 4
            col = dsatur_coloring(conf, m, t0 + time_budget, rng, variant)
            col = local_improve(col, conf, m, t0 + time_budget, rng, max_guard=800)
            k = k_of(col)
            if k < best_num_colors:
                best_num_colors = k
                best_colors = col[:]
        
        # Phase 2: Simulated annealing from best solution
        if best_colors is not None and time.time() < t0 + time_budget:
            rng = Random(seed + 99999)
            sa_col = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng, iterations=300000)
            sa_k = k_of(sa_col)
            if sa_k < best_num_colors:
                best_num_colors = sa_k
                best_colors = sa_col[:]
            sa_col = local_improve(sa_col, conf, m, t0 + time_budget, rng, max_guard=2000)
            sa_k = k_of(sa_col)
            if sa_k < best_num_colors:
                best_num_colors = sa_k
                best_colors = sa_col[:]
            
            if time.time() < t0 + time_budget:
                rng2 = Random(seed + 88888)
                sa_col2 = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng2, iterations=250000)
                sa2_k = k_of(sa_col2)
                if sa2_k < best_num_colors:
                    best_num_colors = sa2_k
                    best_colors = sa_col2[:]
                sa_col2 = local_improve(sa_col2, conf, m, t0 + time_budget, rng2, max_guard=1500)
                sa2_k = k_of(sa_col2)
                if sa2_k < best_num_colors:
                    best_num_colors = sa2_k
                    best_colors = sa_col2[:]
            
            if time.time() < t0 + time_budget - 5:
                rng3 = Random(seed + 77777)
                sa_col3 = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng3, iterations=200000)
                sa3_k = k_of(sa_col3)
                if sa3_k < best_num_colors:
                    best_num_colors = sa3_k
                    best_colors = sa_col3[:]
                sa_col3 = local_improve(sa_col3, conf, m, t0 + time_budget, rng3, max_guard=1500)
                sa3_k = k_of(sa_col3)
                if sa3_k < best_num_colors:
                    best_num_colors = sa3_k
                    best_colors = sa_col3[:]
        
        # Phase 3: Final polish
        if best_colors is not None and time.time() < t0 + time_budget:
            rng = Random(seed + 66666)
            polished = final_polish(best_colors, conf, m, t0 + time_budget, rng)
            pol_k = k_of(polished)
            if pol_k < best_num_colors:
                best_num_colors = pol_k
                best_colors = polished
        
        if best_colors is None:
            best_colors = dsatur_coloring(conf, m, t0 + time_budget)
        
        return best_colors

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    total_time = 115.0
    weights = [n * d for n, d, s in specs]
    total_w = sum(weights)
    budgets = [total_time * w / total_w for w in weights]
    
    out = []
    for i, spec in enumerate(specs):
        if time.time() > t0 + DEADLINE - 5:
            edges = gen_instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        else:
            out.append(solve_one(spec, budgets[i]))
    
    return out
