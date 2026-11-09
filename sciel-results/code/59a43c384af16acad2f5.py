def construct():
    """
    Constructs a C5-free graph with independence number < 9 on 35 vertices.
    Strategy: Disjoint union of 8 cliques.
    - C5-free: Cliques contain no 5-cycles, and disjoint union prevents cycles across components.
    - Independence number: Max independent set picks at most 1 vertex per component.
      With 8 components, alpha = 8 < 9.
    - Maximize n: We use all 35 available vertices distributed among the 8 components.
    """
    n = 35
    edges = []
    
    # Distribute 35 vertices into 8 cliques
    # 6 cliques of size 5, 1 of size 3, 1 of size 2
    # 6*5 + 3 + 2 = 30 + 5 = 35
    component_sizes = [5, 5, 5, 5, 5, 5, 3, 2]
    
    current_vertex = 0
    for size in component_sizes:
        # Add edges for a complete graph (clique) of given size
        for i in range(size):
            for j in range(i + 1, size):
                edges.append([current_vertex + i, current_vertex + j])
        current_vertex += size
        
    return {'n': n, 'edges': edges}
