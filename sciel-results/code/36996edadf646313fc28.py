"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The minimum number of edges for a 3-chromatic 3-uniform hypergraph is 7.
    # A standard example is the Fano plane on 7 vertices.
    # We map the Fano plane vertices to 0..6 and leave 7, 8 isolated.
    
    # Fano plane edges
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for this hypergraph
    # Verified in analysis:
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1
    # Vertices 7, 8 can be any color, e.g., 0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are correct (list of lists for edges, list of ints for colors)
    edges_list = [list(e) for e in edges]
    
    return {'edges': edges_list, 'colors': colors}
# EVOLVE-BLOCK-END
