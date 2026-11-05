def construct():
    import time
    import random
    import sys

    start_time = time.time()
    DEADLINE = 140.0

    def time_up():
        return (time.time() - start_time) > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def greedy_maximal_clique(adj, n, seed_vertex):
        """Grow a maximal clique greedily from seed_vertex."""
        clique = [seed_vertex]
        # Candidates: neighbors of all current clique vertices
        cand_mask = adj[seed_vertex]
        for v in clique[1:]:
            cand_mask &= adj[v]
        # Greedy: pick highest-degree-in-candidates vertex
        while cand_mask:
            if time_up():
                break
            # Pick vertex with max degree in cand_mask
            best_v = -1
            best_deg = -1
            m = cand_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (adj[u] & cand_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = u
                m ^= lsb
            if best_v < 0:
                break
            clique.append(best_v)
            cand_mask &= adj[best_v]
        return clique

    def max_clique_bk(adj, n, active_mask):
        """Bron-Kerbosch with coloring to find max clique in subgraph induced by active_mask."""
        if active_mask == 0:
            return []

        best = [0]
        best_clique = []

        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        deg_in_active = {}
        for v in verts:
            deg_in_active[v] = (adj[v] & active_mask).bit_count()
        verts.sort(key=lambda v: deg_in_active[v], reverse=True)

        def color_sort(P_mask):
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb
            if not verts_p:
                return []
            deg_in_P = {}
            for v in verts_p:
                deg_in_P[v] = (adj[v] & P_mask).bit_count()
            verts_p.sort(key=lambda v: deg_in_P[v], reverse=True)
            v_color = {}
            result = []
            for v in verts_p:
                used_colors = 0
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used_colors |= (1 << v_color[u])
                    nm ^= lsb
                c = 1
                while (used_colors >> c) & 1:
                    c += 1
                v_color[v] = c
                result.append((c, v))
            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best, best_clique
            if P == 0 and X == 0:
                if len(R) > best[0]:
                    best[0] = len(R)
                    best_clique = list(R)
                return
            if time_up():
                return
            colored = color_sort(P)
            for color, v in reversed(colored):
                if len(R) + color <= best[0]:
                    break
                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)
                if time_up():
                    return

        bk([], active_mask, 0)
        return best_clique

    def clique_edge_mask(clique, adj):
        """Return a bitmask representation of edges within clique as a set of (u,v) pairs encoded in a frozenset."""
        edges = set()
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if adj[u] & (1 << v):
                    edges.add((min(u,v), max(u,v)))
        return edges

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        # Total edges
        total_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    total_edges.add((u, v))

        if not total_edges:
            return []

        # Generate candidate cliques
        candidates = []
        seen_cliques = set()

        # 1. Max cliques via Bron-Kerbosch on full graph and subgraphs
        active = (1 << n) - 1
        max_c = max_clique_bk(adj, n, active)
        if max_c and len(max_c) >= 2:
            key = frozenset(max_c)
            if key not in seen_cliques:
                candidates.append(max_c)
                seen_cliques.add(key)

        # 2. Greedy maximal cliques from each vertex as seed
        for v in range(n):
            if time_up():
                break
            cl = greedy_maximal_clique(adj, n, v)
            if len(cl) >= 2:
                key = frozenset(cl)
                if key not in seen_cliques:
                    candidates.append(cl)
                    seen_cliques.add(key)

        # 3. Also try some random subsets for diversity
        rng = random.Random(seed + 1000)
        for _ in range(20):
            if time_up():
                break
            # Random seed vertex
            v = rng.randint(0, n - 1)
            cl = greedy_maximal_clique(adj, n, v)
            if len(cl) >= 2:
                key = frozenset(cl)
                if key not in seen_cliques:
                    candidates.append(cl)
                    seen_cliques.add(key)

        # Precompute edge sets for each candidate
        cand_edges = []
        for cl in candidates:
            es = set()
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    u, v = cl[i], cl[j]
                    if adj[u] & (1 << v):
                        es.add((min(u,v), max(u,v)))
            if es:
                cand_edges.append(es)

        # Greedy set cover: repeatedly pick clique covering most uncovered edges
        uncovered = set(total_edges)
        cliques = []

        while uncovered and not time_up():
            best_idx = -1
            best_count = 0
            for i, es in enumerate(cand_edges):
                cnt = len(es & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_idx < 0 or best_count == 0:
                # No candidate covers any uncovered edge; fallback to individual edges
                break
            cliques.append(candidates[best_idx])
            uncovered -= cand_edges[best_idx]

        # Fallback: cover remaining edges as pairs
        if uncovered:
            for (u, v) in sorted(uncovered):
                cliques.append([u, v])
            uncovered = set()

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
