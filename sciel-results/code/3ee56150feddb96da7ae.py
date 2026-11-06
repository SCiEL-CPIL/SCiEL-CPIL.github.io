"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 20
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            # Color is the smaller vertex index.
            # This ensures that for any color k, the subgraph of edges with color k
            # is a star centered at k (edges {k, v} for v > k).
            # Stars contain no 4-cycles, and edges in a star share a vertex,
            # so no two edges of the same color can be disjoint.
            # Since edges in a 4-cycle are disjoint, no 4-cycle can have
            # two edges of the same color. Thus all 4-cycles are rainbow.
            edges.append([u, v, u])
    return edges
# EVOLVE-BLOCK-END
