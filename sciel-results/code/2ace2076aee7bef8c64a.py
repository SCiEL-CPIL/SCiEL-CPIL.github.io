def construct():
    import time
    start = time.time()
    TOTAL_BUDGET = 110

    from random import Random

    def gen_graph(n, p, seed):
        rng = Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def max_clique_bk(adj, n, deadline, node_limit=300000):
        best = []
        nodes = [0]

        def bk(R, P, X):
            if time.time() > deadline or nodes[0] > node_limit:
                return
            nodes[0] += 1
            if P == 0:
                if X == 0 and len(R) > len(best):
                    best = R[:]
                return
            if len(R) + P.bit_count() <= len(best):
                return
            U = P | X
            if U:
                pivot = -1
                max_deg = -1
                for v in range(n):
                    if (U >> v) & 1:
                        d = (adj[v] & U).bit_count()
                        if d > max_deg:
                            max_deg = d
                            pivot = v
                candidates = P & ~adj[pivot]
            else:
                candidates = P
            for v in range(n):
                if time.time() > deadline or nodes[0] > node_limit:
                    return
                if (candidates >> v) & 1:
                    bk(R + [v], P & adj[v], X & adj[v])
                    P &= ~(1 << v)
                    X |= (1 << v)

        bk([], (1 << n) - 1, 0)
        return best

    def solve_instance(n, p, seed, time_budget):
        adj, edges = gen_graph(n, p, seed)
        if not edges:
            return []
        edge_set = set(edges)
        cliques = []
        deadline = time.time() + time_budget

        while edge_set and time.time() < deadline:
            sub_adj = [0] * n
            for (u, v) in edge_set:
                sub_adj[u] |= (1 << v)
                sub_adj[v] |= (1 << u)
            clique = max_clique_bk(sub_adj, n, deadline, node_limit=200000)
            if len(clique) < 2:
                e = next(iter(edge_set))
                cliques.append(list(e))
                edge_set.discard(e)
            else:
                cliques.append(sorted(clique))
                vs = sorted(clique)
                to_remove = set()
                for i in range(len(vs)):
                    for j in range(i + 1, len(vs)):
                        a, b = vs[i], vs[j]
                        if (a, b) in edge_set:
                            to_remove.add((a, b))
                        elif (b, a) in edge_set:
                            to_remove.add((b, a))
                edge_set -= to_remove

        for e in edge_set:
            cliques.append(list(e))
        return cliques

    cliques1 = solve_instance(32, 0.6, 601, 45)
    elapsed = time.time() - start
    cliques2 = solve_instance(64, 0.55, 602, max(10, TOTAL_BUDGET - elapsed))
    return [cliques1, cliques2]
