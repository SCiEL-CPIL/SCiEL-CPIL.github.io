"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    # The optimal structure is a complete bipartite graph K_8,8.
    # It has 16 vertices, diameter 2, and is triangle-free.
    # Being triangle-free ensures that removing any edge (u, v) increases 
    # the distance between u and v to 3 (via a path of length 3), thus increasing 
    # the graph diameter to 3.
    # Number of edges = 8 * 8 = 64.
    # This is the maximum possible edges for a triangle-free graph on 16 vertices.
    
    edges = []
    # Vertices 0-7 form partition A, 8-15 form partition B
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
