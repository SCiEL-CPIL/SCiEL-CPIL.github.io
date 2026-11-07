"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    from itertools import combinations
    
    deadline = time.time() + 140.0
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Build edge set
        edges = set()
        for u in range(k):
            for v in range(u+1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        
        # Find all maximal cliques
        # For split graphs, maximal cliques are:
        # - For each distinct neighborhood type S (non-empty): S ∪ {I-vertices with that type}
        # - Possibly C itself if no I-vertex has neighborhood C
        # - Single vertices (not useful for edge cover)
        
        # Get I-vertex neighborhoods
        I_nbrs = []
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in edges or (v, u) in edges:
                    nbrs.add(u)
            I_nbrs.append(frozenset(nbrs))
        
        # Group I-vertices
        from collections import defaultdict
        groups = defaultdict(list)
        for i, v in enumerate(range(k, n)):
            groups[I_nbrs[i]].append(v)
        
        # Maximal cliques involving I-vertices
        cliques = []
        for nbr, vertices in groups.items():
            if len(nbr) > 0:
                cliques.append(list(nbr) + vertices)
        
        # Check if C is a maximal clique (no I-vertex has neighborhood = C)
        if frozenset(range(k)) not in groups:
            cliques.append(list(range(k)))
        
        # Now find minimum clique cover using these cliques
        # This is a set cover problem: cover all edges with minimum cliques
        # For small instances, use greedy with local search
        
        # Convert edges to a set for easy lookup
        edge_set = set()
        for e in edges:
            edge_set.add((min(e), max(e)))
        
        # Precompute which edges each clique covers
        clique_edges = []
        for c in cliques:
            ce = set()
            c_set = set(c)
            for u in c:
                for v in c:
                    if u < v:
                        if (u, v) in edge_set:
                            ce.add((u, v))
            clique_edges.append(ce)
        
        # Greedy set cover
        uncovered = set(edge_set)
        used_cliques = []
        available = list(range(len(cliques)))
        
        while uncovered and available:
            best_idx = -1
            best_count = -1
            for idx in available:
                count = len(clique_edges[idx] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1 or best_count == 0:
                break
            used_cliques.append(best_idx)
            uncovered -= clique_edges[best_idx]
            available.remove(best_idx)
        
        # If we still have uncovered edges, we have a problem
        # This shouldn't happen if our clique set is complete
        
        result = [cliques[i] for i in used_cliques]
        # Ensure valid format
        result = [sorted(set(c)) for c in result]
        result = [c for c in result if len(c) >= 2]
        
        return result
    
    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        results.append(solve_split(*spec))
    
    while len(results) < 3:
        spec = specs[len(results)]
        results.append([list(e) for e in instance(*spec)])
    
    return results
# EVOLVE-BLOCK-END
