"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split(n, k, p, seed):
        rng = Random(seed)
        # Generate edges using the same RNG sequence as instance()
        # K-edges: (u, v) for u in range(k), v in range(u+1, k)
        # Cross-edges: for v in range(k, n), for u in range(k): if rng.random() < p: (u, v)
        # We need N(v) for each v in I = range(k, n)
        # The RNG is consumed in the same order: for each v from k to n-1, for each u from 0 to k-1
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs

        cliques = []
        # One clique for the entire K
        if k >= 2:
            cliques.append(list(range(k)))

        # One clique per I-vertex with at least one neighbor
        for v in range(k, n):
            nbrs = neighbors[v]
            if len(nbrs) >= 1:
                clique = [v] + nbrs
                cliques.append(clique)

        # Ensure all vertices are int and lists are proper
        cliques = [[int(x) for x in c] for c in cliques]
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(partition_split(*spec))

    return results
# EVOLVE-BLOCK-END
