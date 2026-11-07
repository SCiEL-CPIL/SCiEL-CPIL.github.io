"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate internal edges of the clique C
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # Generate cross edges between C and I
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return edges, cross_edges

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for n, k, p, seed in specs:
        internal_edges, cross_edges = instance(n, k, p, seed)

        # Build neighbor map for each I-vertex
        neighbors = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            neighbors[v].add(u)

        cliques = []

        # One clique for the entire C (covers all internal edges)
        if k >= 2:
            cliques.append(list(range(k)))

        # For each I-vertex, one clique containing v and all its neighbors in C
        for v in range(k, n):
            N_v = neighbors[v]
            if N_v:
                clique = [v] + sorted(N_v)
                cliques.append(clique)

        # Convert to pure Python int lists
        results.append([[int(x) for x in clique] for clique in cliques])

    return results
# EVOLVE-BLOCK-END
