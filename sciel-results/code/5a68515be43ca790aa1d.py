def construct():
    # Fano plane: 7 edges on 7 vertices, 3-chromatic, 2-coloring impossible.
    # Vertices 7 and 8 are isolated; colors assigned so that every edge is non-monochromatic
    # and the overall coloring uses exactly three colors.
    edges = [
        (0, 1, 3),
        (0, 2, 4),
        (0, 5, 6),
        (1, 2, 5),
        (1, 4, 6),
        (2, 3, 6),
        (3, 4, 5)
    ]
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    return {'edges': edges, 'colors': colors}
