"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))
        
        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))
                    
        return edges_cc, edges_ci

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Fallback to seed-like behavior if running out of time, though it should be fast
            # But we must return valid cliques.
            # The generation is fast, so this shouldn't happen.
            pass
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        cliques = []
        # 1. The clique C itself
        cliques.append(list(range(k)))
        
        # 2. Each cross edge as a clique of size 2
        for u, v in edges_ci:
            cliques.append([u, v])
            
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
