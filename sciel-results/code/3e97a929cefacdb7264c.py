"""Construct a minimal 3-uniform 3-chromatic hypergraph on 9 vertices.

The theoretical minimum number of edges for a 3-chromatic 3-uniform hypergraph
is 7, achieved by the Fano plane. This program returns the Fano plane on
vertices 0-6 with vertices 7 and 8 isolated, along with a valid 3-coloring.
"""

def construct():
    # Fano plane edges on vertices 0..6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    # Valid 3-coloring: vertices 0-6 colored such that no edge is monochromatic.
    # Vertices 7, 8 are isolated, so their color doesn't affect edge validity.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Verify provided coloring is valid (all edges non-monochromatic)
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            raise ValueError("Invalid coloring: monochromatic edge found")

    return {
        'edges': edges,
        'colors': colors
    }
