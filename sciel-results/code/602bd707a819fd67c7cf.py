"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Fano plane on vertices 0-6: 7 edges, 3-uniform, chromatic number 3.
    # Any 3-uniform hypergraph with < 7 edges is 2-colorable (Property B).
    # Vertices 7,8 are isolated; colors assigned to satisfy non-monochromatic edges.
    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )
    colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)
    return {'edges': list(edges), 'colors': list(colors)}
# EVOLVE-BLOCK-END
