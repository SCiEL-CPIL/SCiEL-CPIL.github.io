def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Core edges
        core_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                core_edges.add((u, v))
        
        # Cross edges
        cross_edges = set()
        neighbors = {i: set() for i in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.add((u, v))
                    neighbors[v].add(u)
        
        return core_edges, cross_edges, neighbors

    def solve(n, k, p, seed):
        core_edges, cross_edges, neighbors = instance(n, k, p, seed)
        
        cliques = []
        covered_core_edges = set()
        
        # 1. Add cliques for cross edges
        for i in range(k, n):
            N_i = neighbors[i]
            if N_i:
                # Clique is N_i U {i}
                clique = list(N_i) + [i]
                cliques.append(clique)
                # Mark core edges in N_i as covered
                for u in N_i:
                    for v in N_i:
                        if u < v:
                            covered_core_edges.add((u, v))
        
        # 2. Cover remaining core edges
        remaining_core_edges = core_edges - covered_core_edges
        # Use size-2 cliques for remaining core edges
        for u, v in remaining_core_edges:
            cliques.append([u, v])
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
        
    return results
