"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        # Reconstruct the graph structure
        rng = Random(seed)
        # Build adjacency: for each v in I, find which u in C are connected
        # We need to replicate the exact random calls
        # The instance function does: for v in range(k, n): for u in range(k): rng.random() < p
        # So we replicate that to find neighbors
        neighbors = {}  # v -> list of u in C adjacent to v
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            if nbrs:
                neighbors[v] = nbrs

        cliques = []
        # One clique for the entire clique part C
        if k >= 2:
            cliques.append(list(range(k)))

        # For each v in I with at least one neighbor, create a clique {v} ∪ N(v)
        for v in sorted(neighbors.keys()):
            clique = sorted(neighbors[v] + [v])
            if len(clique) >= 2:
                cliques.append(clique)

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        n, k, p, seed = spec
        cliques = partition_split_graph(n, k, p, seed)
        result.append(cliques)

    # Verify types: all vertices must be Python int, cliques must be lists
    for cliques in result:
        for clique in cliques:
            assert isinstance(clique, list), "clique must be a list"
            for v in clique:
                assert type(v) is int, f"vertex {v} is not Python int"
                assert not isinstance(v, bool), "bool is not allowed"
            assert len(clique) >= 2, "clique must have at least 2 vertices"
            # Check distinct vertices
            assert len(set(clique)) == len(clique), "clique has duplicate vertices"

    return result
# EVOLVE-BLOCK-END
