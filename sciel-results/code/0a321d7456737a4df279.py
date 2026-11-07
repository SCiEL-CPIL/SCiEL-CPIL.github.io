"""
Few hyperedges forcing three colors.

The minimal number of edges in a 3-uniform hypergraph that is 3-chromatic 
(cannot be 2-colored) is 7. The unique such hypergraph (up to isomorphism) 
is the Fano plane. We return the Fano plane on vertices 0..6, with vertices 
7 and 8 isolated. We provide a valid 3-coloring.
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
    # Vertices 7,8 -> color 0 (isolated, any color works)
    colors = [0, 1, 2, 0, 2, 1, 0, 0, 0]

    # Convert to lists of ints as required by type checks
    # edges must be distinct triples on 0..8
    # colors must be nine labels from 0..2
    
    edge_list = [list(e) for e in edges]
    color_list = list(colors)

    # Fast verification that no edge is monochromatic in the provided coloring
    for e in edge_list:
        c0, c1, c2 = color_list[e[0]], color_list[e[1]], color_list[e[2]]
        if c0 == c1 == c2:
            raise ValueError(f"Edge {e} is monochromatic in provided coloring")

    # Fast verification that the hypergraph is NOT 2-colorable.
    # We check all 2^7 = 128 binary assignments for vertices 0..6.
    # If any assignment makes all edges non-monochromatic, it is 2-colorable.
    # For Fano plane, no such assignment exists.
    for mask in range(128):
        # c[i] is the color of vertex i (0 or 1)
        # We can use bit operations for speed
        valid_2coloring = True
        for e in edge_list:
            # Check if all three vertices in e have the same color
            # bit i of mask is color of vertex i
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid_2coloring = False
                break
        if valid_2coloring:
            # This should never happen for the Fano plane
            raise ValueError("Fano plane appears to be 2-colorable, which is impossible")

    return {"edges": edge_list, "colors": color_list}
