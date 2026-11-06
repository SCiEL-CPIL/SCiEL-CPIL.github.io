def construct():
    import time
    from random import Random
    from collections import defaultdict
    import heapq

    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

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

    def find_maximal_induced_matchings(edges, conf, m, deadline):
        """Find a collection of maximal induced matchings (cliques in conflict graph)."""
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        matchings = []
        # Greedy: repeatedly find a maximal induced matching
        used = [False] * m
        remaining = set(range(m))
        
        while remaining and time.time() < deadline:
            # Find a maximal induced matching greedily
            matching = []
            blocked_vertices = set()
            candidates = list(remaining)
            rng_local = Random(len(matchings))
            rng_local.shuffle(candidates)
            
            for e in candidates:
                if e not in remaining:
                    continue
                u, v = edges[e]
                if u in blocked_vertices or v in blocked_vertices:
                    continue
                # Check no edge in matching is adjacent to this one
                ok = True
                for f in matching:
                    if f in conf[e]:
                        ok = False
                        break
                if ok:
                    matching.append(e)
                    # Block all vertices that would prevent future edges
                    # An edge (a,b) blocks any edge sharing a vertex with a or b,
                    # or any edge adjacent to (a,b)
                    for x in (u, v):
                        blocked_vertices.add(x)
                        for y in adj.get(x, set()):
                            blocked_vertices.add(y)
                    remaining.discard(e)
            
            if matching:
                matchings.append(frozenset(matching))
        
        # Also add individual edges as singleton cliques for completeness
        for i in range(m):
            if i not in any(ms for ms in matchings):
                matchings.append(frozenset([i]))
        
        return matchings

    def greedy_clique_cover(matchings, m, deadline):
        """Greedy set-cover: assign each edge to a matching (clique = colour class)."""
        # matchings[j] = frozenset of edge indices that form an induced matching
        # We want to assign each edge to exactly one matching it belongs to,
        # minimizing the number of matchings used.
        
        # Build: for each edge, which matchings contain it
        edge_to_cliques = defaultdict(list)
        for j, ms in enumerate(matchings):
            for e in ms:
                edge_to_cliques[e].append(j)
        
        # Greedy: repeatedly pick the clique that covers the most uncovered edges
        uncovered = set(range(m))
        assignment = [-1] * m
        used_cliques = []
        
        while uncovered and time.time() < deadline:
            best_clique = -1
            best_count = 0
            for j, ms in enumerate(matchings):
                if j in used_cliques:
                    continue
                cnt = len(ms & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_clique = j
            
            if best_clique < 0 or best_count == 0:
                # No clique covers any uncovered edge; assign remaining edges to new colours
                break
            
            ms = matchings[best_clique]
            for e in ms:
                if e in uncovered:
                    assignment[e] = len(used_cliques)
                    uncovered.discard(e)
            used_cliques.append(best_clique)
        
        # Assign any remaining uncovered edges to new colours
        next_c = len(used_cliques)
        for e in sorted(uncovered):
            assignment[e] = next_c
            next_c += 1
        
        return assignment

    def merge_clique_cover(assignment, conf, m, deadline):
        """Try to merge colour classes (cliques) to reduce total count."""
        col = list(assignment)
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < 200:
            improved = False
            guard += 1
            cls = defaultdict(set)
            for i, c in enumerate(col):
                cls[c].add(i)
            
            cols = sorted(cls.keys())
            merged = False
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
                        improved = True
                        break
                if merged:
                    break
            if not merged:
                break
        return col

    def dsatur_coloring(conf, m, deadline):
        """DSATUR-style coloring: prioritize edges with most colored conflicting neighbors."""
        col = [-1] * m
        forbidden = [set() for _ in range(m)]
        remaining = set(range(m))
        
        while remaining and time.time() < deadline:
            best_edge = None
            best_count = -1
            for e in remaining:
                cnt = len(forbidden[e])
                if cnt > best_count:
                    best_count = cnt
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
        return max(col) + 1 if col else 0

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
        """Check if edge e can be assigned colour new_c."""
        for j in conf[e]:
            if col[j] == new_c:
                return False
        return True

    def local_improve(col, conf, m, deadline, rng):
        col = list(col)
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < 500:
            improved = False
            guard += 1
            cls = build_classes(col)
            
            # --- merge pass: fold one whole class into another when compatible ---
            while time.time() < deadline:
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
                            improved = True
                            break
                    if merged:
                        break
                if not merged:
                    break

            # --- single-edge moves: try to move an edge to any valid lower colour ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    moved = False
                    for _ in range(5000):
                        e = rng.randrange(m)
                        c = col[e]
                        for new_c in range(c):
                            if is_valid_move(col, e, new_c, conf):
                                col[e] = new_c
                                moved = True
                                break
                        if moved:
                            break
                    if moved:
                        improved = True

            # --- 2-swap moves ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    swapped = False
                    for _ in range(3000):
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

            # --- 3-swap moves ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 3:
                    tri_swapped = False
                    for _ in range(2000):
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
                            tri_swapped = True
                            break
                    if tri_swapped:
                        improved = True

        return reindex(col)

    def simulated_annealing(col, conf, m, deadline, rng, iterations=120000):
        col = list(col)
        best_col = col[:]
        best_k = k_of(col)
        T = 1.0
        cool = 0.99998
        max_c = best_k
        # Track colour counts for O(1) k computation
        color_counts = [0] * (max_c + 2)
        for c in col:
            if c < len(color_counts):
                color_counts[c] += 1
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            e = rng.randrange(m)
            c_old = col[e]
            r = rng.random()
            if r < 0.5:
                new_c = rng.randrange(c_old) if c_old > 0 else 0
            else:
                new_c = rng.randrange(max_c + 2)
            
            if new_c == c_old:
                continue
            
            valid = True
            for j in conf[e]:
                if col[j] == new_c:
                    valid = False
                    break
            
            if valid:
                col[e] = new_c
                # Update counts
                color_counts[c_old] -= 1
                if new_c >= len(color_counts):
                    color_counts.extend([0] * (new_c + 2 - len(color_counts)))
                color_counts[new_c] += 1
                
                # Compute new k
                new_k = best_k
                if color_counts[best_k - 1] == 0 and best_k > 0:
                    new_k = best_k - 1
                    while new_k > 0 and color_counts[new_k - 1] == 0:
                        new_k -= 1
                if new_c + 1 > new_k:
                    new_k = new_c + 1
                
                if new_k < best_k:
                    best_col = col[:]
                    best_k = new_k
                    T = max(0.01, T * 0.99)
                else:
                    delta = new_k - best_k
                    if delta == 0:
                        if rng.random() < 0.4:
                            pass
                        else:
                            col[e] = c_old
                            color_counts[c_old] += 1
                            color_counts[new_c] -= 1
                    else:
                        if rng.random() < (2.718281828 ** (-delta / T)):
                            pass
                        else:
                            col[e] = c_old
                            color_counts[c_old] += 1
                            color_counts[new_c] -= 1
                T = max(0.01, T * cool)
        
        return best_col

    def solve_one(spec, time_budget):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conf = build_conflict(edges)
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Phase 0: Clique-cover approach (new)
        if time.time() < t0 + time_budget:
            matchings = find_maximal_induced_matchings(edges, conf, m, t0 + time_budget)
            if matchings:
                cc_col = greedy_clique_cover(matchings, m, t0 + time_budget)
                cc_col = merge_clique_cover(cc_col, conf, m, t0 + time_budget)
                cc_col = reindex(cc_col)
                cc_k = k_of(cc_col)
                if cc_k < best_num_colors:
                    best_num_colors = cc_k
                    best_colors = cc_col[:]
                # Polish the clique cover result
                rng_cc = Random(seed + 55555)
                polished_cc = local_improve(cc_col, conf, m, t0 + time_budget, rng_cc)
                pk_cc = k_of(polished_cc)
                if pk_cc < best_num_colors:
                    best_num_colors = pk_cc
                    best_colors = polished_cc[:]
        
        # Phase 1: DSATUR + local improvement
        for trial in range(8):
            if time.time() > t0 + time_budget:
                break
            rng = Random(seed + 1000 * trial)
            col = dsatur_coloring(conf, m, t0 + time_budget)
            col = local_improve(col, conf, m, t0 + time_budget, rng)
            k = k_of(col)
            if k < best_num_colors:
                best_num_colors = k
                best_colors = col[:]
        
        # Phase 2: Simulated annealing from best solution with multiple restarts
        if best_colors is not None and time.time() < t0 + time_budget:
            for sa_trial in range(3):
                if time.time() > t0 + time_budget:
                    break
                rng = Random(seed + 99999 + 1000 * sa_trial)
                sa_col = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng, iterations=100000)
                sa_k = k_of(sa_col)
                if sa_k < best_num_colors:
                    best_num_colors = sa_k
                    best_colors = sa_col[:]
                sa_col = local_improve(sa_col, conf, m, t0 + time_budget, rng)
                sa_k = k_of(sa_col)
                if sa_k < best_num_colors:
                    best_num_colors = sa_k
                    best_colors = sa_col[:]
        
        # Phase 3: Final polish
        if best_colors is not None and time.time() < t0 + time_budget:
            rng = Random(seed + 77777)
            polished = local_improve(best_colors, conf, m, t0 + time_budget, rng)
            pk = k_of(polished)
            if pk < best_num_colors:
                best_num_colors = pk
                best_colors = polished
        
        if best_colors is None:
            best_colors = dsatur_coloring(conf, m, t0 + time_budget)
        
        return best_colors

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    total_time = 110.0
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
