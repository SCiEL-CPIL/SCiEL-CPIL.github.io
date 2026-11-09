"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from itertools import combinations

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot selection
        if P or X:
            # Choose pivot u from P ∪ X with max degree in P
            pivot_set = P | X
            pivot = max(pivot_set, key=lambda u: len(adj[u] & P))
            candidates = P - adj[pivot]
        else:
            candidates = P

        for v in list(candidates):
            bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
            P.remove(v)
            X.add(v)

    def find_maximal_cliques(adj):
        n = len(adj)
        cliques = []
        bron_kerbosch(adj, frozenset(), frozenset(range(n)), frozenset(), cliques)
        return cliques

    def greedy_set_cover(cliques, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        uncovered = set(range(len(cliques)))
        cover = []
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                cnt = sum(1 for i in uncovered if v in cliques[i])
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_count == 0:
                # Should not happen if cliques are non-empty
                break
            cover.append(best_v)
            uncovered = {i for i in uncovered if best_v not in cliques[i]}
        return cover

    def overlap_graph_maximal_cliques(cliques):
        """
        Build overlap graph: vertex i represents clique i,
        edge (i,j) if cliques[i] & cliques[j] != empty.
        Find maximal cliques in this overlap graph.
        For each maximal clique in overlap graph, the intersection
        of all original cliques in it is a hitting set candidate.
        """
        m = len(cliques)
        if m == 0:
            return []
        # Build adjacency for overlap graph
        overlap_adj = [set() for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if cliques[i] & cliques[j]:
                    overlap_adj[i].add(j)
                    overlap_adj[j].add(i)

        # Bron-Kerbosch on overlap graph
        result_cliques = []
        def bk(R, P, X):
            if not P and not X:
                if R:
                    result_cliques.append(frozenset(R))
                return
            if P or X:
                pivot_set = P | X
                pivot = max(pivot_set, key=lambda u: len(overlap_adj[u] & P))
                candidates = P - overlap_adj[pivot]
            else:
                candidates = P
            for v in list(candidates):
                bk(R | {v}, P & overlap_adj[v], X & overlap_adj[v])
                P.remove(v)
                X.add(v)
        bk(frozenset(), frozenset(range(m)), frozenset())
        return result_cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = find_maximal_cliques(adj)
        # Filter to size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        if not cliques:
            return []

        # Candidate 1: greedy set cover
        greedy_cover = greedy_set_cover(cliques, n)

        # Candidate 2: from overlap graph maximal cliques
        best_cover = greedy_cover
        try:
            overlap_mcliques = overlap_graph_maximal_cliques(cliques)
            for oc in overlap_mcliques:
                # Intersection of all cliques in this overlap clique
                common = set(cliques[list(oc)[0]])
                for idx in oc:
                    common &= set(cliques[idx])
                if common:
                    candidate = list(common)
                    # Verify it hits all cliques
                    valid = all(any(v in cliques[i] for v in candidate) for i in range(len(cliques)))
                    if valid and len(candidate) < len(best_cover):
                        best_cover = candidate
        except Exception:
            pass

        # Also try: for each vertex, check if it alone hits all cliques
        for v in range(n):
            if all(v in c for c in cliques):
                return [v]

        return best_cover

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        result = solve_instance(n, p, seed)
        out.append(sorted(result))
    return out
# EVOLVE-BLOCK-END
