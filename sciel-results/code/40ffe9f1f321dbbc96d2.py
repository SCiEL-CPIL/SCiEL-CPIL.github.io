"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    deadline = time.time() + 140.0  # safety margin before the 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build the exact same split graph as the seed generator.
        # Clique on first k vertices, independent set on the rest.
        # Cross pairs joined with probability p.
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def build_partition(cross_edges, n, k):
        """
        Valid clique partition minimizing the number of cliques:
        - One big clique on the first k vertices: covers every edge inside K_k
          exactly once (all K_k edges live in this single clique).
        - For each independent-set vertex v with at least one cross neighbour,
          the set {v} union N(v) is a clique (N(v) is a subset of the clique
          vertices, which are mutually adjacent). It covers every cross edge
          incident to v exactly once, and no K_k edge is double-covered because
          the big clique is the only clique covering K_k edges.
        Every edge lies in exactly one listed clique; every clique has >= 2
        vertices (the big clique has k>=2, and each per-v clique has v plus
        at least one neighbour).
        Total = 1 + (number of independent vertices with >= 1 neighbour).
        """
        # Map each independent vertex to the set of its clique neighbours.
        neigh = {v: [] for v in range(k, n)}
        for (u, v) in cross_edges:
            neigh[v].append(int(u))

        cliques = [list(range(k))]  # big clique covering all K_k edges
        for v in range(k, n):
            ns = neigh[v]
            if len(ns) >= 1:
                # {v} union its neighbours is a clique.
                cliques.append([int(v)] + [int(x) for x in ns])
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > deadline:
            # Fallback identical to the main path (computation is tiny).
            ce = instance(n, k, p, seed)
            result.append(build_partition(ce, n, k))
        else:
            ce = instance(n, k, p, seed)
            result.append(build_partition(ce, n, k))

    # Ensure all types are Python int and lists.
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
# EVOLVE-BLOCK-END
