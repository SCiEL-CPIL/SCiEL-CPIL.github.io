"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct a complete bipartite graph K(10,10) with vertices partitioned into L={0..9} and R={10..19}
    # Color each edge (u, v) with c = (u + (v - 10)) % 20
    # This ensures that every 4-cycle in the graph is rainbow (all four edge colors distinct)
    
    edges = []
    L = list(range(0, 10))
    R = list(range(10, 20))
    
    for u in L:
        for v in R:
            j = v - 10
            c = (u + j) % 20
            edges.append([u, v, c])
    
    return edges
# EVOLVE-BLOCK-END
