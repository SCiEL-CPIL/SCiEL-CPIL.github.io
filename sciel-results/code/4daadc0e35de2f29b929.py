"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_3,3 components.
    
    The graph is formed by two independent bipartite components:
    1. Vertices {0, 1, 2} connected to {3, 4, 5}
    2. Vertices {6, 7, 8} connected to {9, 10, 11}
    
    This ensures:
    - Every vertex has degree exactly 3.
    - The graph is triangle-free (girth 4).
    - The only cycle lengths present are 4 and 6.
    - The number of distinct cycle lengths is 2, which is the theoretical minimum 
      for any graph with minimum degree at least 3.
    """
    edges = []
    
    # Define the two bipartite partitions for the first component
    comp1_left = (0, 1, 2)
    comp1_right = (3, 4, 5)
    
    # Define the two bipartite partitions for the second component
    comp2_left = (6, 7, 8)
    comp2_right = (9, 10, 11)
    
    # Generate edges for the first K_3,3 component
    for u in comp1_left:
        for v in comp1_right:
            edges.append([int(u), int(v)])
            
    # Generate edges for the second K_3,3 component
    for u in comp2_left:
        for v in comp2_right:
            edges.append([int(u), int(v)])
            
    return edges

# EVOLVE-BLOCK-END
