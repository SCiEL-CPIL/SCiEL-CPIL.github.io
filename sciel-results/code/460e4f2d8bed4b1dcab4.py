"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        return sorted(edges + cross)
    
    def solve_instance(n, k, p, seed):
        """Find minimum clique partition for a split graph."""
        rng = Random(seed)
        
        # Build adjacency for cross edges
        # neighbors[v] = set of u in K connected to v (v in I)
        neighbors = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            neighbors[v] = nbrs
        
        cliques = []
        
        # Clique 1: all of K (covers all K-K edges)
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each I vertex with neighbors, create clique N(v) ∪ {v}
        for v in range(k, n):
            nbrs = neighbors[v]
            if len(nbrs) >= 1:
                clique = list(nbrs) + [v]
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
    
    # Ensure types are correct: all vertices are Python int, cliques are lists
    final_result = []
    for cliques in results:
        clean_cliques = []
        for clique in cliques:
            clean_clique = [int(v) for v in clique]
            if len(clean_clique) >= 2:
                clean_cliques.append(clean_clique)
        final_result.append(clean_cliques)
    
    return final_result
# EVOLVE-BLOCK-END
