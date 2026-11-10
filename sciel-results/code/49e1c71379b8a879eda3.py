"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    
    for n, k, p, seed in specs:
        if time.time() > deadline - 5.0:
            # Fallback to seed-like behavior if running out of time (unlikely)
            edges = instance(n, k, p, seed)
            result.append([list(e) for e in edges])
            continue
            
        edges = instance(n, k, p, seed)
        
        cliques = []
        
        # 1. Add the clique K
        cliques.append(list(range(k)))
        
        # 2. Add individual cross edges as cliques
        # We need to distinguish K-K edges from K-I edges
        # K-K edges: u < k and v < k
        # K-I edges: u < k and v >= k
        for u, v in edges:
            if u < k and v >= k:
                cliques.append([u, v])
        
        result.append(cliques)
        
    return result
# EVOLVE-BLOCK-END
