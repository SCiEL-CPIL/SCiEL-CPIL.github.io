"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    deadline = time.time() + 110  # safety margin before 120s limit

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_cliques(n, k, p, seed):
        # Generate edges using the same logic as instance
        rng = Random(seed)
        # Build neighbor sets for vertices in the independent set [k, n)
        # For each vertex v in [k, n), find which vertices in [0, k) it connects to
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            if nbrs:
                neighbors[v] = nbrs

        cliques = []
        # The clique on the first k vertices
        cliques.append(list(range(k)))
        # For each vertex in the independent set that has neighbors in C,
        # the set N_C(v) ∪ {v} is a clique (since C is a clique and v connects to all in N_C(v))
        for v in range(k, n):
            if v in neighbors:
                cliques.append(sorted(neighbors[v] + [v]))

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = partition_cliques(n, k, p, seed)
        # Convert to plain Python ints and lists
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)

    return results
# EVOLVE-BLOCK-END
