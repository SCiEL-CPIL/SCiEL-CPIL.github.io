import time
import itertools

def construct():
    """
    Return a 3-uniform hypergraph on 9 vertices (0..8) with exactly 3 chromatic number,
    minimizing the number of edges. The Fano plane (7 edges on 7 vertices) is the
    unique minimal such hypergraph; we embed it on vertices 0..6 and leave 7,8 isolated.
    """
    deadline = time.time() + 55  # safety margin before 90s kill

    # Fano plane edges on vertices 0..6
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    # A valid 3-coloring of the Fano plane:
    # v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1
    # Verify each edge is non-monochromatic:
    # (0,1,2): 0,1,2 OK
    # (0,3,4): 0,1,2 OK
    # (0,5,6): 0,0,1 OK
    # (1,3,5): 1,1,0 OK
    # (1,4,6): 1,2,1 OK
    # (2,3,6): 2,1,1 OK
    # (2,4,5): 2,2,0 OK
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Convert to required types
    edges = [list(e) for e in fano_edges]

    return {
        'edges': edges,
        'colors': colors
    }
