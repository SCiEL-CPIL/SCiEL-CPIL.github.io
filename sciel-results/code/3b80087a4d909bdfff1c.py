"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 16
    
    # 1. Pre-compute Q3 edge patterns on vertices 0..7
    # Canonical Q3 edges
    q3_edges_canonical = []
    for i in range(8):
        for j in range(i+1, 8):
            if (i ^ j) in (1, 2, 4):
                q3_edges_canonical.append((i, j))
        
    # Generate all unique isomorphic edge sets
    patterns = set()
    for perm in itertools.permutations(range(8)):
        new_edges = set()
        for u, v in q3_edges_canonical:
            a, b = perm[u], perm[v]
            if a > b: a, b = b, a
            new_edges.add((a, b))
        patterns.add(frozenset(new_edges))
    
    # Convert to list for iteration
    pattern_list = list(patterns)
    
    # 2. Pre-compute all 8-vertex subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    
    # 3. Initialize complete graph
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            edges.add((u, v))
            
    # Helper to calculate degrees
    def get_degrees(edge_set):
        degs = {i: 0 for i in range(n)}
        for u, v in edge_set:
            degs[u] += 1
            degs[v] += 1
        return degs

    # 4. Greedy removal loop
    # We iterate until no Q3 is found
    # To speed up, we can shuffle subsets or use a fixed order
    # Let's use a fixed order but break early
    
    while True:
        q3_found = False
        best_edge_to_remove = None
        best_score = -1
        current_degs = get_degrees(edges)
        
        for subset in all_subsets:
            # Build induced edge set (indices 0..7)
            induced_edges = set()
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = subset[i], subset[j]
                    if u > v: u, v = v, u
                    if (u, v) in edges:
                        induced_edges.add((i, j))
            
            if len(induced_edges) < 12:
                continue
            
            # Check if any pattern is a subset of induced_edges
            for p in pattern_list:
                if p.issubset(induced_edges):
                    # Found a Q3
                    # Map pattern edges back to graph vertices
                    q3_actual_edges = []
                    for i, j in p:
                        u, v = subset[i], subset[j]
                        if u > v: u, v = v, u
                        q3_actual_edges.append((u, v))
                    
                    # Select best edge to remove
                    # Heuristic: Max degree sum in current graph
                    for e in q3_actual_edges:
                        u, v = e
                        score = current_degs[u] + current_degs[v]
                        if score > best_score:
                            best_score = score
                            best_edge_to_remove = e
                    q3_found = True
                    break # Break pattern loop
                
                if q3_found:
                    break # Break subset loop
            
            if q3_found:
                break # Break subset loop
        
        if not q3_found:
            break
        
        edges.remove(best_edge_to_remove)
        
    return [list(e) for e in edges]
# EVOLVE-BLOCK-END
