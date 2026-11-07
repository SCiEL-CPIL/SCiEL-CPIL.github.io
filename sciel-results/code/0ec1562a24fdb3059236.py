import time
import random

def construct():
    deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def max_clique_tomita(vertices, adj):
        """
        Find a maximum clique in a graph defined by 'vertices' (list of nodes)
        and 'adj' (dict of sets of neighbors in the subgraph).
        Uses Tomita's algorithm with pivoting.
        """
        if not vertices:
            return []

        best_clique = []

        def expand(P, X):
            nonlocal best_clique
            if not P:
                if len(X) > len(best_clique):
                    best_clique = list(X)
                return

            if len(X) + len(P) <= len(best_clique):
                return

            # Pivoting: pick vertex v with max degree in P
            # P = P1 ∪ {v} ∪ P2 where P1 = neighbors of v in P, P2 = non-neighbors
            v = max(P, key=lambda x: len(adj[x] & P))
            P1 = P & adj[v]
            P2 = P - adj[v] - {v}

            # Expand P2 first (non-neighbors of v), then P1
            for u in list(P2):
                if len(X) + len(P) <= len(best_clique):
                    return
                new_P = P & adj[u]
                new_X = X | {u}
                expand(new_P, new_X)
                P.remove(u)

            # Now expand P1 (neighbors of v)
            for u in list(P1):
                if len(X) + len(P) <= len(best_clique):
                    return
                new_P = P & adj[u]
                new_X = X | {u}
                expand(new_P, new_X)
                P.remove(u)

        expand(set(vertices), set())
        return best_clique

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # Build full adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        uncovered = set(edge_set)
        cliques = []

        while uncovered:
            if time.time() > deadline - 5.0:
                # Fallback: cover remaining edges with pairwise cliques
                for u, v in list(uncovered):
                    if (u, v) in uncovered or (v, u) in uncovered:
                        cliques.append([u, v])
                        uncovered.discard((u, v))
                        uncovered.discard((v, u))
                break

            # Build graph of uncovered edges
            # Vertices: all vertices incident to uncovered edges
            # Edges: uncovered edges
            verts = set()
            for u, v in uncovered:
                verts.add(u)
                verts.add(v)

            if not verts:
                break

            # Build adjacency for the subgraph induced by uncovered edges
            sub_adj = {v: set() for v in verts}
            for u, v in uncovered:
                sub_adj[u].add(v)
                sub_adj[v].add(u)

            # Find maximum clique in this subgraph
            clique = max_clique_tomita(list(verts), sub_adj)

            if not clique:
                # Should not happen if uncovered is non-empty
                # Fallback: pick any edge
                u, v = next(iter(uncovered))
                clique = [u, v]

            # Add clique to result
            cliques.append(clique)

            # Remove all edges within this clique from uncovered
            clique_set = set(clique)
            to_remove = set()
            for u in clique:
                for v in clique:
                    if u < v:
                        to_remove.add((u, v))
                    elif u > v:
                        to_remove.add((v, u))
            uncovered -= to_remove

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10.0:
            # Emergency fallback for remaining instances
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [[u, v] for u, v in edges]
            result.append(cliques)
            continue

        cliques = solve_split(*spec)
        result.append(cliques)

    return result
