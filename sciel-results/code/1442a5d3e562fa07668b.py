"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    # Construct K_{8,8} which is the maximum edge count triangle-free graph on 16 vertices.
    # This graph has diameter 2.
    # Removing any edge (u, v) where u in A and v in B:
    # dist(u, v) becomes 3 because N(u) and N(v) are disjoint in the remaining graph 
    # (N(u) = B\{v}, N(v) = A\{u}, intersection empty).
    # Thus diameter increases to 3.
    # Edge count is 8 * 8 = 64.
    
    edges = []
    for i in range(8):
        for j in range(8, 16):
            edges.append([i, j])
    return edges
# EVOLVE-BLOCK-END
