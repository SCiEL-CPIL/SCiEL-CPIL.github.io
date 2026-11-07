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

    def _min_clique_cover_c(adj, k):
        """Minimum clique cover of a graph on k vertices (0..k-1) with adjacency adj.
        = min coloring of complement. Uses DSATUR greedy."""
        if k <= 1:
            return []
        comp = [set(range(k)) for _ in range(k)]
        for s in range(k):
            comp[s].discard(s)
            for w in adj[s]:
                comp[s].discard(w)
        
        colors = [-1] * k
        sat_colors = [set() for _ in range(k)]
        degree = [len(comp[s]) for s in range(k)]
        uncolored = set(range(k))
        
        for _ in range(k):
            best_v = None
            best_key = None
            for v in uncolored:
                key = (len(sat_colors[v]), degree[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v
            uncolored.discard(best_v)
            c = 0
            while c in sat_colors[best_v]:
                c += 1
            colors[best_v] = c
            for w in comp[best_v]:
                if w in uncolored and c not in sat_colors[w]:
                    sat_colors[w].add(c)
        
        max_c = max(colors) if colors else 0
        cliques = [[] for _ in range(max_c + 1)]
        for s in range(k):
            cliques[colors[s]].append(s)
        return [c for c in cliques if len(c) >= 2]

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        N_C = {}
        for v in range(k, n):
            N_C[v] = set(u for u in range(k) if (u, v) in edge_set)

        cross_edges_norm = []
        for (a, b) in edge_set:
            if a < k and b >= k:
                cross_edges_norm.append((a, b))
            elif b < k and a >= k:
                cross_edges_norm.append((b, a))

        base_cliques = [list(range(k))]
        for (u, v) in cross_edges_norm:
            base_cliques.append(sorted([u, v]))
        best_cliques = base_cliques
        best_count = len(base_cliques)

        all_cc_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        uncovered_cc = set(all_cc_edges)
        uncovered_cross = set(cross_edges_norm)

        v_cc = {}
        v_cross = {}
        for v in range(k, n):
            nc = N_C[v]
            nc_list = sorted(nc)
            cc_v = set((a, b) for i, a in enumerate(nc_list) for b in nc_list[i + 1:])
            v_cc[v] = cc_v
            cross_v = set((u, v) for u in nc)
            v_cross[v] = cross_v

        selected = []
        S_v = {}

        while (uncovered_cc or uncovered_cross) and not _time_up():
            best_v = None
            best_gain = 0
            for v in range(k, n):
                if v in selected:
                    continue
                gain = len(uncovered_cc & v_cc[v]) + len(uncovered_cross & v_cross[v])
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v is None or best_gain == 0:
                break
            selected.append(best_v)
            S_v[best_v] = set(N_C[best_v])
            uncovered_cc -= v_cc[best_v]
            uncovered_cross -= v_cross[best_v]

        cliques = []
        for v in selected:
            sv = S_v[v]
            if sv:
                cliques.append(sorted([v] + list(sv)))

        if uncovered_cc:
            involved = set()
            for (a, b) in uncovered_cc:
                involved.add(a)
                involved.add(b)
            inv_list = sorted(involved)
            idx = {v: i for i, v in enumerate(inv_list)}
            m = len(inv_list)
            sub_adj = [set() for _ in range(m)]
            for (a, b) in uncovered_cc:
                ia, ib = idx[a], idx[b]
                sub_adj[ia].add(ib)
                sub_adj[ib].add(ia)
            rem_cliques = _min_clique_cover_c(sub_adj, m)
            for rc in rem_cliques:
                rc_orig = [inv_list[i] for i in rc]
                if len(rc_orig) >= 2:
                    cliques.append(sorted(rc_orig))

        if len(cliques) == 0:
            return best_cliques, best_count

        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return best_cliques, best_count

        edge_cover = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover[(a, b)] = edge_cover.get((a, b), 0) + 1

        for e in edge_set:
            if edge_cover.get(e, 0) != 1:
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
