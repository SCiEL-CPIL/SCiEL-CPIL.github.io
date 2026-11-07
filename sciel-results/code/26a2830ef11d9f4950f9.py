"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth >= 4.
    The graph consists of two disjoint copies of K3,3.
    K3,3 is 3-regular, bipartite (girth 4), and has cycle lengths {4, 6}.
    The set of distinct cycle lengths for the whole graph is {4, 6}, so the raw value is 2.
    """
    edges = []
    
    # First K3,3 component on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K3,3 component on vertices 6..11
    # Partition C: 6, 7, 8
    # Partition D: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
