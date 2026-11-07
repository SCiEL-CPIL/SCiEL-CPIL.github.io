import time
from random import Random
from collections import deque

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _dsatur_color_bipartite(c_adj, v_adj, num_c, num_v):
        """Color bipartite graph: C vertices 0..num_c-1, V vertices num_c..num_c+num_v-1.
        Returns color list (length num_c + num_v) using DSATUR greedy."""
        total = num_c + num_v
        adj = [set() for _ in range(total)]
        for c in range(num_c):
            for v in c_adj[c]:
                adj[c].add(num_c + v)
                adj[num_c + v].add(c)
        for v in range(num_v):
            for c in v_adj[v]:
                adj[num_c + v].add(c)
                adj[c].add(num_c + v)

        colors = [-1] * total
        sat_colors = [set() for _ in range(total)]
        degree = [len(adj[i]) for i in range(total)]
        uncolored = set(range(total))

        for _ in range(total):
            best_node = None
            best_key = None
            for node in uncolored:
                key = (len(sat_colors[node]), degree[node])
                if best_key is None or key > best_key:
                    best_key = key
                    best_node = node
            uncolored.discard(best_node)

            c = 0
            while c in sat_colors[best_node]:
                c += 1
            colors[best_node] = c

            for nb in adj[best_node]:
                if nb in uncolored and c not in sat_colors[nb]:
                    sat_colors[nb].add(c)

        return colors

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # N_C[v] = set of C-vertices adjacent to I-vertex v
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        cross_edges_norm = []
        for (a, b) in edge_set:
            if a < k and b >= k:
                cross_edges_norm.append((a, b))
            elif b < k and a >= k:
                cross_edges_norm.append((b, a))

        # Regime 1 baseline: 1 C-clique + per-cross-edge 2-cliques
        base_cliques = [list(range(k))]
        for (u, v) in cross_edges_norm:
            base_cliques.append(sorted([u, v]))
        best_cliques = base_cliques
        best_count = len(base_cliques)

        # Regime 2: Select I-vertices to cover all C-C edges, then color bipartite graph
        all_cc_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        total_cc = len(all_cc_edges)

        if total_cc == 0:
            return best_cliques, best_count

        # For each I-vertex, which C-C edges can it cover?
        # Edge (u,w) covered by v iff u in N_C[v] and w in N_C[v]
        # Represent as: for each v, the set of C-vertices it's adjacent to
        # C-C edges covered by v = all pairs within N_C[v]

        # Greedy set cover: select I-vertices to cover all C-C edges
        uncovered = set(all_cc_edges)
        selected_v = []
        V_v = {v: set() for v in range(k, n)}

        # Precompute for each v: set of C-C edges it covers
        v_cover = {}
        for v in range(k, n):
            nc = sorted(N_C[v])
            if len(nc) >= 2:
                v_cover[v] = set((nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc)))
            else:
                v_cover[v] = set()

        while uncovered and not _time_up():
            best_v = None
            best_cnt = 0
            for v in range(k, n):
                if v in selected_v:
                    continue
                cnt = len(uncovered & v_cover[v])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None or best_cnt == 0:
                break
            selected_v.append(best_v)
            uncovered -= v_cover[best_v]

        if uncovered:
            # Can't cover all C-C edges with selected I-vertices
            # Try adding more or fall back
            return best_cliques, best_count

        # Now: for each selected v, determine V_v[v] (set of C-vertices in its clique)
        # Constraint: for each cross edge (u, v), u must be in V_v[v]
        # Constraint: for each C-C edge (u,w), exists selected v with u,w in V_v[v]
        # V_v[v] ⊆ N_C[v]

        # Initialize V_v[v] with cross-edge requirements
        for (u, v) in cross_edges_norm:
            if v in selected_v:
                V_v[v].add(u)

        # Now we need to ensure all C-C edges are covered.
        # For each C-C edge (u,w), find a selected v with u,w in N_C[v], and add both to V_v[v]
        # To minimize cliques, we want to assign each C-vertex to as few I-vertices as possible.
        # This is equivalent to: partition C-vertices into groups, each group assigned to one I-vertex.
        # But each C-vertex can appear in multiple I-vertex cliques (it just creates more cliques).
        #
        # Actually, the key insight: the bipartite graph between C and selected V vertices
        # has edges (c, v) if c in N_C[v]. We need to assign each C-vertex to one or more
        # selected V-vertices such that:
        # 1. Every cross edge (c, v) is covered: c assigned to v
        # 2. Every C-C edge (c1, c2) is covered: exists selected v with c1,c2 both assigned to v
        #
        # The number of cliques = sum over selected v of (1 if V_v[v] non-empty else 0)
        # But wait - each selected v contributes exactly 1 clique if V_v[v] is non-empty.
        # So total cliques = number of selected v with non-empty V_v[v].
        #
        # Since we need cross edges covered, every selected v that has cross edges must have
        # non-empty V_v[v]. So total cliques = |selected_v| (assuming all have cross edges).
        #
        # But we might be able to do better: not all C-C edges need to be covered by a single
        # I-vertex. We can have multiple I-vertices each covering some C-C edges.
        # The constraint is: for each C-C edge (c1,c2), there must be at least one selected v
        # with c1,c2 in V_v[v].
        #
        # To minimize total cliques, we want to minimize |selected_v| (since each contributes 1).
        # The greedy set cover already minimizes |selected_v|.
        #
        # But we also need to verify that the cross-edge constraints are satisfiable.
        # For cross edge (u,v): u must be in V_v[v]. Since u in N_C[v] (edge exists), this is fine.
        #
        # However, we also need: for each C-C edge (u,w), exists v in selected_v with u,w in V_v[v].
        # The greedy set cover ensured this is possible, but we need to actually assign.
        #
        # Strategy: for each C-C edge not yet covered, find a selected v that can cover it
        # (u,w in N_C[v]) and add u,w to V_v[v].
        #
        # But this might make V_v[v] too large, creating invalid cliques? No - since C is complete,
        # any subset of C is a clique. So {v} ∪ V_v[v] is always a clique as long as V_v[v] ⊆ N_C[v].

        # Check which C-C edges are already covered by cross-edge assignments
        cc_covered = set()
        for v in selected_v:
            nc_v = sorted(V_v[v])
            for i in range(len(nc_v)):
                for j in range(i + 1, len(nc_v)):
                    cc_covered.add((nc_v[i], nc_v[j]))

        remaining_cc = set(all_cc_edges) - cc_covered

        # For each remaining C-C edge, assign to a selected v that can cover it
        for (u, w) in sorted(remaining_cc):
            if _time_up():
                break
            # Find a selected v with u,w in N_C[v]
            assigned = False
            for v in selected_v:
                if u in N_C[v] and w in N_C[v]:
                    V_v[v].add(u)
                    V_v[v].add(w)
                    assigned = True
                    break
            if not assigned:
                # Should not happen if greedy set cover succeeded
                return best_cliques, best_count

        # Now build cliques
        cliques = []
        for v in selected_v:
            if V_v[v]:
                clique = sorted([v] + list(V_v[v]))
                cliques.append(clique)

        # Verify validity
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return best_cliques, best_count

        # Verify all edges covered exactly once
        edge_cover_count = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover_count[(a, b)] = edge_cover_count.get((a, b), 0) + 1

        for e in edge_set:
            if edge_cover_count.get(e, 0) != 1:
                return best_cliques, best_count

        if len(cliques) < best_count:
            best_cliques = cliques
            best_count = len(cliques)

        return best_cliques, best_count

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append(sorted([u, v]))
            results.append(cliques)
            continue
        n, k, p, seed = spec
        cliques, _ = _solve_instance(n, k, p, seed)
        results.append(cliques)
    return results
