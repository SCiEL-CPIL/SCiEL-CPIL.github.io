"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges), cross_edges

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        c_edges, cross_edges = instance(n, k, p, seed)
        
        cliques = []
        
        # 1. Add the entire clique C as a single clique.
        # This covers all internal edges in C exactly once.
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. For each cross edge (u, v), add a 2-clique [u, v].
        # This covers each cross edge exactly once.
        for u, v in cross_edges:
            cliques.append([u, v])
            
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
