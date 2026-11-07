"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Construct two disjoint K_3,3 components
    # Component 1: Vertices 0-5, bipartition {0,1,2} and {3,4,5}
    # Component 2: Vertices 6-11, bipartition {6,7,8} and {9,10,11}
    
    edges = []
    
    # Helper to add complete bipartite edges
    def add_bipartite_edges(u_set, v_set):
        for u in u_set:
            for v in v_set:
                edges.append([u, v])
                
    # First K_3,3
    u1 = [0, 1, 2]
    v1 = [3, 4, 5]
    add_bipartite_edges(u1, v1)
    
    # Second K_3,3
    u2 = [6, 7, 8]
    v2 = [9, 10, 11]
    add_bipartite_edges(u2, v2)
    
    return edges

# EVOLVE-BLOCK-END
