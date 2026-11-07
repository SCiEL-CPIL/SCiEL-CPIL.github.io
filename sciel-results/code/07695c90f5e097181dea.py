"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency: for each independent vertex v, list of its neighbors in C
        cross_neighbors = {}
        for v in range(k, n):
            neighbors = [u for u in range(k) if rng.random() < p]
            cross_neighbors[v] = neighbors
        return cross_neighbors
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cross_neighbors = instance(n, k, p, seed)
        
        cliques = []
        # Clique for all K-edges: the entire clique part
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each independent vertex with at least one cross edge, add clique {v} ∪ N(v)
        for v in range(k, n):
            neighbors = cross_neighbors[v]
            if len(neighbors) >= 1:
                clique = [v] + list(neighbors)
                cliques.append(clique)
        
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
