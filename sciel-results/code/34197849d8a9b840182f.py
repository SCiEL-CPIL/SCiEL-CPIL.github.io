"""
Few hyperedges forcing three colors.

We return the Fano plane (7 edges on 7 of the 9 vertices), which is the
edge-minimal 3-uniform hypergraph that is 3-chromatic.  Vertices 7 and 8 are
isolated.  A valid 3-coloring is provided; every edge is non-monochromatic.

Construction: directly map PG(2,2) lines to triples.
Verification: exhaustively check all 2^7 binary colorings to confirm no 2-coloring exists.
"""

def construct():
    # Fano plane edges (triples) on vertices 0..6
    # These are the 7 lines of the projective plane PG(2,2)
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]

    # A 3-coloring: each edge must contain at least two distinct colors.
    # Known valid coloring for Fano plane:
    # vertices 0,3,6 -> color 0
    # vertices 1,5 -> color 1
    # vertices 2,4 -> color 2
    colors = [0, 1, 2, 0, 2, 1, 0, 0, 0]

    # Verify the coloring is valid (no monochromatic edges)
    for e in edges:
        assert colors[e[0]] != colors[e[1]] or colors[e[1]] != colors[e[2]] or colors[e[0]] != colors[e[2]], \
            f"Edge {e} is monochromatic with colors {colors[e[0]]}, {colors[e[1]]}, {colors[e[2]]}"

    # Verify the hypergraph is NOT 2-colorable by checking all 2^7 = 128 binary colorings
    # If any 2-coloring makes all edges non-monochromatic, the graph would be 2-chromatic
    # For the Fano plane, no such coloring exists (it is 3-chromatic)
    for mask in range(128):
        c = [(mask >> i) & 1 for i in range(7)]
        valid_2coloring = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                valid_2coloring = False
                break
        if valid_2coloring:
            # This should never happen for the Fano plane
            # If it does, we have a bug in our edge set
            raise ValueError("Fano plane appears to be 2-colorable, which is impossible")

    return {"edges": [list(e) for e in edges], "colors": list(colors)}
