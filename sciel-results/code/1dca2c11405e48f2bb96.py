"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_one(n, k, p, seed, deadline):
        edges = instance(n, k, p, seed)
        if time.time() > deadline:
            return [list(e) for e in edges]
        
        # Build edge set
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        # Identify clique vertices (0..k-1) and independent set vertices (k..n-1)
        clique_verts = set(range(k))
        
        # Build all maximal cliques:
        # 1. The full clique C = {0, ..., k-1}
        # 2. For each i in independent set, {i} ∪ N(i) where N(i) ⊆ C
        max_cliques = []
        
        # Clique C
        C = list(range(k))
        C_edges = set()
        for idx, u in enumerate(C):
            for v in C[idx+1:]:
                C_edges.add((min(u,v), max(u,v)))
        max_cliques.append((C, C_edges))
        
        # For each vertex in independent set
        for i in range(k, n):
            # Find neighbors in C
            nbrs = []
            for u in range(k):
                if (min(i,u), max(i,u)) in edge_set:
                    nbrs.append(u)
            if not nbrs:
                continue
            cliq = [i] + sorted(nbrs)
            # Edges in this clique
            cliq_edges = set()
            for idx, u in enumerate(cliq):
                for v in cliq[idx+1:]:
                    cliq_edges.add((min(u,v), max(u,v)))
            max_cliques.append((cliq, cliq_edges))
        
        # Greedy exact cover: repeatedly pick clique covering most uncovered edges
        uncovered = set(edge_set)
        result = []
        
        while uncovered:
            if time.time() > deadline:
                # Fallback: cover remaining edges individually
                for e in uncovered:
                    result.append([e[0], e[1]])
                break
            
            best_clique = None
            best_count = -1
            for cliq, cliq_edges in max_cliques:
                count = len(cliq_edges & uncovered)
                if count > best_count:
                    best_count = count
                    best_clique = (cliq, cliq_edges)
            
            if best_count <= 0:
                # Should not happen if max cliques cover all edges
                for e in uncovered:
                    result.append([e[0], e[1]])
                break
            
            result.append(list(best_clique[0]))
            uncovered -= best_clique[1]
        
        return result
    
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback for remaining
            for remaining_spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(results):]:
                edges = instance(*remaining_spec)
                results.append([list(e) for e in edges])
            break
        results.append(solve_one(*spec, deadline))
    
    return results
# EVOLVE-BLOCK-END
