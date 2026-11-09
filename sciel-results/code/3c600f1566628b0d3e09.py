"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the Fano plane, the unique 3-uniform 3-chromatic hypergraph on 7 vertices.
    It has exactly 7 edges and is the minimum-edge 3-uniform hypergraph that is
    not 2-colorable.  Vertices 7 and 8 are isolated.

    A valid 3-coloring is provided; every edge is non-monochromatic under it.
    """
    # Fano-plane edges as a list of 3-tuples of Python ints on vertices 0..6
    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )

    # Nine colour labels (0, 1 or 2) for vertices 0..8.
    # 0:0  1:1  2:2  3:1  4:2  5:0  6:1  7:0  8:0
    colours = (0, 1, 2, 1, 2, 0, 1, 0, 0)

    return {'edges': edges, 'colors': colours}

# EVOLVE-BLOCK-END
