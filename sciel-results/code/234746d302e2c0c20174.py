def construct():
    import time
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve(n, k, p, seed, deadline):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)

        # available core edges (as a set of frozensets of size 2 for fast membership)
        available = set()
        for u in range(k):
            for v in range(u + 1, k):
                available.add((u, v))

        cliques = []

        # Group I-vertices by identical neighbor set
        groups = {}
        for i in range(k, n):
            key = frozenset(adj[i])
            groups.setdefault(key, []).append(i)

        # Process each group: pack core vertices into maximal cliques
        for neighbors, members in groups.items():
            # Build adjacency among core vertices restricted to 'neighbors'
            # A core vertex v is "available" if v in neighbors
            N = neighbors
            if not N:
                continue
            # adjacency within N
            adjN = {v: set() for v in N}
            for v in N:
                for w in N:
                    if v < w and (v, w) in available:
                        adjN[v].add(w)
                        adjN[w].add(v)

            # Greedy maximal clique packing
            remaining = set(N)
            while remaining:
                if time.time() > deadline:
                    # Fallback: cover each remaining vertex as a 2-clique with first member
                    for v in sorted(remaining):
                        cliques.append([v] + sorted(members))
                    break
                # Greedy maximal clique: pick vertices in order, add if connected to all chosen
                S = set()
                for v in sorted(remaining):
                    ok = True
                    for u in S:
                        if v not in adjN[u]:
                            ok = False
                            break
                    if ok:
                        S.add(v)
                if not S:
                    v = min(remaining)
                    S = {v}
                cliques.append(sorted(members) + sorted(S))
                # Remove all core edges inside S from available
                for u in S:
                    for w in S:
                        if u < w:
                            available.discard((u, w))
                remaining -= S

        # Cover any remaining core edges with pure core cliques
        remaining_edges = set(available)
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(sorted([e[0], e[1]]))
                break
            # Build adjacency on remaining edges
            adj_rem = {v: set() for v in range(k)}
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            # Greedy maximal clique
            S = set()
            for v in range(k):
                if not adj_rem[v]:
                    continue
                ok = True
                for u in S:
                    if v not in adj_rem[u]:
                        ok = False
                        break
                if ok:
                    S.add(v)
            if len(S) < 2:
                u, v = min(remaining_edges)
                S = {u, v}
            cliques.append(sorted(S))
            for u in S:
                for w in S:
                    if u < w:
                        remaining_edges.discard((u, w))

        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 120

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve(*spec, deadline))

    final_results = []
    for cliques in results:
        clean_cliques = []
        for clique in cliques:
            clean_cliques.append([int(v) for v in clique])
        final_results.append(clean_cliques)

    return final_results
