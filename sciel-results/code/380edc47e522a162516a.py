"""Rewritten: structural clique-partition via max-bipartite-matching bicliques + greedy max-clique + local refinement."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    _T0 = time.time()
    _DEADLINE = 150.0 - 12.0  # safety margin

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

    def _max_clique(adj, cand):
        # Bron–Kerbosch with pivot, returns a maximum clique (list of ints)
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            # pivot
            U = P | X
            if not U:
                piv = None
            else:
                piv = max(U, key=lambda v: len(adj[v] & P))
            cand = [v for v in P if v not in adj[piv]] if piv is not None else list(P)
            for v in cand:
                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P.remove(v)
                X.add(v)
        bk([], set(cand), set())
        return best

    def _karp_hopcroft(adj, L, R):
        # max cardinality bipartite matching; returns dict L->R
        # adj: dict v -> set(neighbors)
        from collections import deque
        matchL = {}
        matchR = {}
        def bfs():
            dist = {u: 0 for u in L}
            q = deque([u for u in L if u not in matchL])
            found = False
            while q:
                u = q.popleft()
                for w in adj[u]:
                    mw = matchR.get(w)
                    if mw is None:
                        found = True
                    elif mw not in dist:
                        dist[mw] = dist[u] + 1
                        q.append(mw)
            return dist, found
        def dfs(u):
            for w in adj[u]:
                mw = matchR.get(w)
                if mw is None or (mw in dist and dist[mw] == dist[u] + 1 and dfs(mw)):
                    matchL[u] = w
                    matchR[w] = u
                    return True
            dist[u] = None
            return False
        while True:
            dist, found = bfs()
            if not found:
                break
            for u in L:
                if u not in matchL:
                    dfs(u)
        return matchL

    def _solve_one(n, k, p, seed):
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))            # clique side
        I = list(range(k, n))         # independent side
        # cross adjacency
        cross = {u: {v for v in adj[u] if v >= k} for u in C}

        # --- Phase 1: build large bicliques from the cross bipartite graph ---
        cliques = []
        covered = set()               # covered edges (frozenset of 2)

        # internal clique edge set
        internal_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                internal_edges.add((i, j))

        # We'll repeatedly find a max matching in the residual cross graph,
        # then extract bicliques (common-neighborhood cliques) from it.
        residual_cross = {u: set(cross[u]) for u in C}

        def add_clique(verts):
            nonlocal covered
            vs = sorted(verts)
            if len(vs) < 2:
                return
            new_edges = []
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    e = (vs[i], vs[j])
                    if e not in covered:
                        new_edges.append(e)
            if not new_edges:
                return
            # verify it's a clique
            if not _is_clique(adj, vs):
                return
            for e in new_edges:
                covered.add(e)
            cliques.append(vs)

        # Phase 1a: seed with the full clique C (covers all internal edges)
        add_clique(C)

        # Phase 1b: iteratively find bicliques in residual cross graph
        # A biclique on A (subset of C) and B (subset of I) requires all
        # cross edges between A and B present. We grow via matchings.
        for _ in range(40):
            if time.time() > _DEADLINE:
                break
            # residual cross adjacency
            res = {u: set() for u in C}
            for u in C:
                for v in cross[u]:
                    if (u, v) not in covered:
                        res[u].add(v)
            L = [u for u in C if res[u]]
            R = set()
            for u in L:
                R |= res[u]
            R = list(R)
            if not L or not R:
                break
            # Hopcroft-Karp
            adjB = {u: set(res[u]) for u in L}
            matchL = _karp_hopcroft(adjB, L, R)
            if not matchL:
                break
            # From the matching, build candidate bicliques:
            # For each matched pair (u, v), the biclique = {u} U N(v) within res,
            # i.e. all u' in L adjacent to v. Grow greedily.
            progress = False
            for u, v in matchL.items():
                if (u, v) in covered:
                    continue
                # biclique: A = all u' in C with v in res[u']; B = all v' in R with u in res[v']... 
                # simpler: A = {u' : v in res[u']}, then B = common neighbors of A
                A = [u2 for u2 in C if v in res[u2]]
                if len(A) < 2:
                    A = [u]
                # common neighbors of A
                common = set(res[A[0]])
                for u2 in A[1:]:
                    common &= res[u2]
                B = list(common)
                if not B:
                    continue
                verts = A + B
                # ensure every cross edge between A and B is uncovered & present
                ok = True
                for u2 in A:
                    for v2 in B:
                        if (u2, v2) in covered or (u2, v2) not in adj[u2]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    # shrink B to just matched
                    B = [v]
                    verts = A + B
                    ok = True
                    for u2 in A:
                        if (u2, v) in covered or (u2, v) not in adj[u2]:
                            ok = False
                    if not ok:
                        continue
                add_clique(verts)
                progress = True
            if not progress:
                break

        # --- Phase 2: greedy max-clique on residual edges ---
        # residual edges
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))
        residual_edges = [e for e in all_edges if e not in covered]

        # build residual adjacency
        res_adj = {v: set() for v in range(n)}
        for (u, v) in residual_edges:
            res_adj[u].add(v)
            res_adj[v].add(u)

        # repeatedly take max clique on the vertex with max residual degree
        while residual_edges and time.time() < _DEADLINE:
            # pick vertex of max residual degree
            start = max(range(n), key=lambda v: len(res_adj[v]))
            if len(res_adj[start]) < 1:
                # isolated residual edges left: pair them up
                rem = [e for e in residual_edges if e not in covered]
                for (u, v) in rem:
                    if (u, v) not in covered:
                        add_clique([u, v])
                break
            cand = {start} | res_adj[start]
            mc = _max_clique(res_adj, cand)
            if len(mc) < 2:
                # fallback: edge
                rem = [e for e in residual_edges if e not in covered]
                for (u, v) in rem:
                    if (u, v) not in covered:
                        add_clique([u, v])
                break
            before = len(covered)
            add_clique(mc)
            if len(covered) == before:
                # no progress; pair off
                rem = [e for e in residual_edges if e not in covered]
                for (u, v) in rem:
                    if (u, v) not in covered:
                        add_clique([u, v])
                break
            # update res_adj
            for v in mc:
                res_adj[v].clear()
            for (u, v) in residual_edges:
                if u in mc or v in mc:
                    continue
                if v in res_adj[u]:
                    res_adj[u].add(v)
                    res_adj[v].add(u)

        # safety: cover any remaining edges individually
        for (u, v) in all_edges:
            if (u, v) not in covered:
                add_clique([u, v])

        # --- Phase 3: local refinement (merge/replace to reduce count) ---
        # Try to merge two cliques if their union is a clique.
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    uni = set(cliques[i]) | set(cliques[j])
                    if len(uni) >= 2 and _is_clique(adj, uni):
                        # check union covers exactly the same edges as i+j
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

        # final validation
        for cl in cliques:
            assert len(cl) >= 2
            assert len(set(cl)) == len(cl)
            assert _is_clique(adj, cl)
        # every edge in exactly one clique
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
            # fallback: one clique per edge
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
