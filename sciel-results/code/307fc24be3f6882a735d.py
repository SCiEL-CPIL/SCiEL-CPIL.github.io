"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        if time.time() > deadline:
            # Fallback: just return per-edge cliques (valid but not optimal)
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in sorted(edges)]

        edges = instance(n, k, p, seed)

        # Find cross-edges: (u, v) where u < k <= v
        # For each independent vertex v >= k, find its neighbors in the clique
        cross_neighbors = {}  # v -> list of u's
        for (u, v) in edges:
            if u < k and v >= k:
                if v not in cross_neighbors:
                    cross_neighbors[v] = []
                cross_neighbors[v].append(u)

        cliques = []
        # One big clique for the clique part (vertices 0..k-1)
        if k >= 2:
            cliques.append(list(range(k)))

        # For each independent vertex with cross-edges, one clique
        for v in sorted(cross_neighbors.keys()):
            neighbors = cross_neighbors[v]
            if len(neighbors) >= 1:
                clique = [v] + sorted(neighbors)
                if len(clique) >= 2:
                    cliques.append(clique)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback for remaining instances
            for s in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(results):]:
                edges = instance(*s)
                results.append([[u, v] for u, v in sorted(edges)])
            break
        results.append(partition_cliques(*spec))

    # Ensure all values are Python int and lists are lists
    final = []
    for clique_list in results:
        cleaned = []
        for clique in clique_list:
            cleaned.append([int(v) for v in clique])
        final.append(cleaned)

    return final
# EVOLVE-BLOCK-END
