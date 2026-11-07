"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a dense graph on 16 vertices with diameter exactly 2,
    where every edge is essential (deleting any single edge increases
    diameter above 2 or disconnects the graph).
    
    The optimal structure is the complete bipartite graph K_{8,8},
    which has 64 edges. This is the unique extremal triangle-free graph
    on 16 vertices, and any triangle would make an edge non-essential
    (since the third vertex provides a length-2 path).
    """
    n = 16
    edges = []
    # Partition vertices into two sets of 8
    # A = 0..7, B = 8..15
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
