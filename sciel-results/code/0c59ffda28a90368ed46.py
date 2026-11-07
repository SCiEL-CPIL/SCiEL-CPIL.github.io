"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        m = len(edges)
        if m == 0:
            return []

        # adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # candidate cliques: the clique side C, and {v} U N(v) for independent v
        C = list(range(k))
        cand = [C]
        for v in range(k, n):
            cand.append([v] + sorted(adj[v]))

        # edge id per unordered pair
        eid = {}
        for i, (u, v) in enumerate(edges):
            eid[(u, v)] = i

        # edges covered by each candidate clique
        cand_edges = []
        for clique in cand:
            s = set()
            L = len(clique)
            for a in range(L):
                ua = clique[a]
                for b in range(a + 1, L):
                    vb = clique[b]
                    if ua < vb:
                        idx = eid.get((ua, vb))
                    else:
                        idx = eid.get((vb, ua))
                    if idx is not None:
                        s.add(idx)
            cand_edges.append(s)

        # greedy set cover over edges
        remaining = set(range(m))
        chosen = []
        while remaining:
            best_i = -1
            best_cov = -1
            best_size = None
            for i, es in enumerate(cand_edges):
                cov = len(remaining & es)
                if cov == 0:
                    continue
                if cov > best_cov or (cov == best_cov and (best_size is None or len(cand[i]) < best_size)):
                    best_cov = cov
                    best_i = i
                    best_size = len(cand[i])
            if best_i < 0:
                # should not happen; fallback to single edges
                for idx in sorted(remaining):
                    u, v = edges[idx]
                    chosen.append([u, v])
                break
            chosen.append(list(cand[best_i]))
            remaining -= cand_edges[best_i]

        # verify exact partition: every edge in exactly one clique
        count = [0] * m
        for clique in chosen:
            L = len(clique)
            for a in range(L):
                ua = clique[a]
                for b in range(a + 1, L):
                    vb = clique[b]
                    if ua < vb:
                        idx = eid.get((ua, vb))
                    else:
                        idx = eid.get((vb, ua))
                    if idx is None:
                        return None  # non-edge inside clique -> invalid
                    count[idx] += 1
        for c in count:
            if c != 1:
                return None  # edge uncovered or double covered -> invalid
        return chosen

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve(*spec)
        if cliques is None:
            # safety fallback: one clique per edge (always valid)
            cliques = [list(e) for e in instance(*spec)]
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
