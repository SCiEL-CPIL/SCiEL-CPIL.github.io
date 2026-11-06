def construct():
    """
    Construct a C5-free graph with independence number < 9.
    Uses 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    Alpha = 8 * 1 = 8 < 9.
    """
    total_vertices = 32
    graph_edges = []
    
    # Create 8 independent cliques of size 4
    for clique_index in range(8):
        start_vertex = clique_index * 4
        clique_vertices = [start_vertex + k for k in range(4)]
        
        # Add all edges within this clique
        for u_idx in range(4):
            for v_idx in range(u_idx + 1, 4):
                graph_edges.append([clique_vertices[u_idx], clique_vertices[v_idx]])
                
    return {'n': total_vertices, 'edges': graph_edges}
