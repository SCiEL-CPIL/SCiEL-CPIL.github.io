import time

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum possible number of edges (7 edges, Fano plane).

    The Fano plane is the unique (up to isomorphism) minimum 3-chromatic
    3-uniform hypergraph. Any 3-uniform hypergraph with fewer than 7 edges
    is 2-colorable.

    We define the Fano plane on vertices 0..6 and leave vertices 7 and 8 isolated.
    """
    deadline = time.time() + 85  # safety margin before 90s kill

    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    )
    colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)

    # Ensure all values are plain Python int (not bool, not numpy int)
    edges = tuple(tuple(int(v) for v in e) for e in edges)
    colors = tuple(int(c) for c in colors)

    return {
        'edges': edges,
        'colors': colors
    }
