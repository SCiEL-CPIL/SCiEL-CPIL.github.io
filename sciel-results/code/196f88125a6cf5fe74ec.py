"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        edges += cross_edges
        return edges

    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Determine neighbors in C for each vertex in I
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs

        cliques = []
        
        # 1. Cover all C-I edges with 2-vertex cliques {v, u}
        # This ensures no C-C edges are covered by these cliques.
        for v in range(k, n):
            for u in neighbors[v]:
                cliques.append([v, u])
                
        # 2. Cover all C-C edges with one large clique C
        # Since C is a complete graph, this is a valid clique.
        # It covers all C-C edges exactly once.
        # It does not cover any C-I edges (as it contains no I vertices).
        if k >= 2:
            cliques.append(list(range(k)))
            
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        result.append(solve_split(*spec))

    return result
# EVOLVE-BLOCK-END
