"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 16
    # Construct G as complete bipartite K_8,8
    # This corresponds to complement G_bar being two disjoint cliques of size 8
    # Vertices 0..7 in Clique 1, 8..15 in Clique 2
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
