def construct():
    import sys
    sys.setrecursionlimit(100000)

    def generate_edges(n, p, seed):
        # Reproduce verifier's random_graph edge set using random.Random(seed)
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal(R, P, X, adj):
        """Yield maximal cliques (as sets of vertices) within subgraph induced by adj-restricted set."""
        if not P and not X:
            yield R
            return
        # pivot selection: choose u in P|X maximizing |P & adj[u]|
        pivot_set = P | X
        if pivot_set:
            best_u = None
            best_count = -1
            for u in pivot_set:
                c = len(P & adj.get(u, set()))
                if c > best_count:
                    best_count = c
                    best_u = u
            candidates = P - adj.get(best_u, set())
        else:
            candidates = P
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj.get(v, set())
            new_X = X & adj.get(v, set())
            for clique in bron_kerbosch_maximal(new_R, new_P, new_X, adj):
                yield clique
            P.remove(v)
            X.add(v)

    def solve(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []

        # adjacency as sets
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # track uncovered edges
        uncovered = set(edges)

        def edges_of_clique(vertices):
            vs = list(vertices)
            es = []
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, b = vs[i], vs[j]
                    if a > b:
                        a, b = b, a
                    es.append((a, b))
            return es

        result = []

        while uncovered:
            # Find the largest clique whose all edges are uncovered.
            # Candidate vertex set = vertices incident to uncovered edges.
            cand_vertices = set()
            for u, v in uncovered:
                cand_vertices.add(u)
                cand_vertices.add(v)

            # Build subgraph adjacency restricted to uncovered edges among cand_vertices
            sub_adj = {v: set() for v in cand_vertices}
            for u, v in uncovered:
                sub_adj[u].add(v)
                sub_adj[v].add(u)

            # Find all maximal cliques in this subgraph and track the largest
            best_clique = None
            best_size = 1

            # Run Bron-Kerbosch on the entire subgraph to find the true maximum clique
            # This is more thorough than the seeded approach
            for clique in bron_kerbosch_maximal(set(), cand_vertices, set(), sub_adj):
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique

            if best_clique is None or len(best_clique) < 2:
                # Fallback: take a single edge
                e = next(iter(uncovered))
                best_clique = {e[0], e[1]}

            # Remove all edges internal to best_clique
            cl_edges = edges_of_clique(best_clique)
            for e in cl_edges:
                uncovered.discard(e)

            # Only add if it actually covers >= 1 edge (clique size>=2 ensures that)
            result.append(sorted(best_clique))

        return result

    return [solve(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
