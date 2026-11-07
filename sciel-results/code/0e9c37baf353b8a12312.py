"""Improved: greedy max-clique set-cover with aggressive local search refinement for split graphs."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import deque

    _T0 = time.time()
    _DEADLINE = 150.0 - 10.0  # safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _is_clique(adj, verts):
        vs = list(verts)
        for i in range(len(vs)):
            a = vs[i]
            for j in range(i + 1, len(vs)):
                if vs[j] not in adj[a]:
                    return False
        return True

    def _max_clique_bk(adj, cand):
        """Bron-Kerbosch with pivot; returns maximum clique from candidate set."""
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            U = P | X
            if not U:
                piv = None
            else:
                piv = max(U, key=lambda v: len(adj[v] & P))
            candidates = [v for v in P if v not in adj[piv]] if piv is not None else list(P)
            for v in candidates:
                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P.remove(v)
                X.add(v)
        bk([], set(cand), set())
        return best

    def _solve_one(n, k, p, seed):
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))

        cliques = []
        covered = set()

        def add_clique(verts):
            nonlocal covered
            vs = sorted(verts)
            if len(vs) < 2:
                return False
            new_edges = []
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    e = (vs[i], vs[j])
                    if e not in covered:
                        new_edges.append(e)
            if not new_edges:
                return False
            if not _is_clique(adj, vs):
                return False
            for e in new_edges:
                covered.add(e)
            cliques.append(vs)
            return True

        # Phase 1: Greedy max-clique set cover
        # Build residual adjacency
        res_adj = {v: set() for v in range(n)}
        for (u, v) in all_edges:
            res_adj[u].add(v)
            res_adj[v].add(u)

        # First, always take the full K-clique (it's the largest clique in a split graph)
        add_clique(list(range(k)))
        for v in range(k):
            res_adj[v] = set()
        for (u, v) in all_edges:
            if u >= k and v >= k:
                pass  # no I-I edges
            elif u < k and v >= k:
                # already handled by clearing res_adj for K vertices
                pass

        # Rebuild res_adj properly after removing K-K edges
        res_adj = {v: set() for v in range(n)}
        remaining = set(all_edges) - covered
        for (u, v) in remaining:
            res_adj[u].add(v)
            res_adj[v].add(u)

        # Greedy: repeatedly pick max clique in residual graph
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > _DEADLINE:
                break
            # Find vertex with max residual degree
            max_deg = 0
            start_v = -1
            for v in range(n):
                d = len(res_adj[v])
                if d > max_deg:
                    max_deg = d
                    start_v = v
            if max_deg == 0:
                break
            if max_deg == 1:
                # Only isolated edges left
                rem = list(remaining)
                for (u, v) in rem:
                    add_clique([u, v])
                remaining = set()
                break

            # Find max clique containing start_v in residual graph
            cand = {start_v} | res_adj[start_v]
            mc = _max_clique_bk(res_adj, cand)
            if len(mc) < 2:
                mc = [start_v] + list(res_adj[start_v])[:1]

            before = len(covered)
            add_clique(mc)
            if len(covered) == before:
                # No progress, just pair off remaining
                rem = list(remaining)
                for (u, v) in rem:
                    add_clique([u, v])
                remaining = set()
                break

            # Update residual
            mc_set = set(mc)
            new_remaining = set()
            for (u, v) in remaining:
                if u in mc_set or v in mc_set:
                    continue
                new_remaining.add((u, v))
            remaining = new_remaining
            # Update res_adj
            for v in range(n):
                if v in mc_set:
                    res_adj[v] = set()
            for (u, v) in remaining:
                res_adj[u].add(v)
                res_adj[v].add(u)

        # Safety: cover any remaining
        for (u, v) in remaining:
            add_clique([u, v])
        remaining = set()

        # Phase 2: Local refinement - try to merge cliques whose union is a clique
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            for i in range(len(cliques)):
                if time.time() > _DEADLINE - 5:
                    break
                for j in range(i + 1, len(cliques)):
                    uni = set(cliques[i]) | set(cliques[j])
                    if len(uni) < 3:
                        continue
                    if not _is_clique(adj, uni):
                        continue
                    # Check: edges of i and j must exactly equal edges of union
                    def edge_set(cl):
                        s = set()
                        for a in range(len(cl)):
                            for b in range(a + 1, len(cl)):
                                s.add((cl[a], cl[b]))
                        return s
                    ei = edge_set(cliques[i])
                    ej = edge_set(cliques[j])
                    eu = edge_set(list(uni))
                    if ei | ej == eu:
                        cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                        cliques.append(sorted(uni))
                        improved = True
                        break
                if improved:
                    break

        # Phase 3: Try to absorb vertices from smaller cliques into larger ones
        improved = True
        while improved and time.time() < _DEADLINE - 3:
            improved = False
            for i in range(len(cliques)):
                if time.time() > _DEADLINE - 3:
                    break
                for j in range(len(cliques)):
                    if i == j:
                        continue
                    # Try adding each vertex of cliques[j] to cliques[i]
                    new_cl = list(cliques[i])
                    for v in cliques[j]:
                        if v not in new_cl:
                            candidate = new_cl + [v]
                            if _is_clique(adj, candidate):
                                new_cl = candidate
                    if len(new_cl) > len(cliques[i]):
                        # Check edge consistency: new_cl edges must be subset of
                        # (cliques[i] edges) + (cliques[j] edges) + (edges between them)
                        def edge_set(cl):
                            s = set()
                            for a in range(len(cl)):
                                for b in range(a + 1, len(cl)):
                                    s.add((cl[a], cl[b]))
                            return s
                        ei = edge_set(cliques[i])
                        ej = edge_set(cliques[j])
                        en = edge_set(new_cl)
                        # All edges in new_cl must be in ei or ej
                        if en <= (ei | ej):
                            # Remove vertices from cliques[j] that are now in new_cl
                            removed = set(new_cl) - set(cliques[i])
                            remaining_j = [v for v in cliques[j] if v not in removed]
                            if len(remaining_j) >= 2:
                                cliques[i] = sorted(new_cl)
                                cliques[j] = sorted(remaining_j)
                            else:
                                cliques[i] = sorted(new_cl)
                                cliques = [c for idx, c in enumerate(cliques) if idx != j]
                            improved = True
                            break
                if improved:
                    break

        # Phase 4: Final merge pass
        improved = True
        while improved and time.time() < _DEADLINE - 2:
            improved = False
            for i in range(len(cliques)):
                if time.time() > _DEADLINE - 2:
                    break
                for j in range(i + 1, len(cliques)):
                    uni = set(cliques[i]) | set(cliques[j])
                    if len(uni) < 3:
                        continue
                    if not _is_clique(adj, uni):
                        continue
                    def edge_set(cl):
                        s = set()
                        for a in range(len(cl)):
                            for b in range(a + 1, len(cl)):
                                s.add((cl[a], cl[b]))
                        return s
                    ei = edge_set(cliques[i])
                    ej = edge_set(cliques[j])
                    eu = edge_set(list(uni))
                    if ei | ej == eu:
                        cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                        cliques.append(sorted(uni))
                        improved = True
                        break
                if improved:
                    break

        # Final validation
        for cl in cliques:
            assert len(cl) >= 2
            assert len(set(cl)) == len(cl)
            assert _is_clique(adj, cl)
        seen = set()
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (cl[a], cl[b])
                    assert e not in seen, "edge in two cliques"
                    seen.add(e)
        assert seen == all_edges, "edge not covered"

        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
