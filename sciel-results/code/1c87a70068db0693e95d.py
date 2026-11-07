import time
import itertools

def _verify_three_chromatic(edges, colors):
    """Verify that the hypergraph is not 2-colorable and the provided coloring is valid."""
    # Check the provided coloring is valid (no monochromatic edges)
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            raise ValueError("Provided coloring is invalid: monochromatic edge found")
    
    # Check that no 2-coloring exists
    # For a 3-uniform hypergraph, a 2-coloring fails if for every assignment of {0,1} to vertices,
    # at least one edge is monochromatic.
    # We check all 2^9 = 512 possible 2-colorings.
    for bits in range(1 << 9):
        assignment = [(bits >> i) & 1 for i in range(9)]
        valid = True
        for e in edges:
            a, b, c = assignment[e[0]], assignment[e[1]], assignment[e[2]]
            if a == b == c:
                valid = False
                break
        if valid:
            return False  # Found a valid 2-coloring, so it's 2-colorable
    
    return True  # No 2-coloring found, so it's 3-chromatic

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices (0..8) that is 
    3-chromatic (not 2-colorable) with the minimum number of edges.
    
    The minimum number of edges for a 3-uniform non-2-colorable hypergraph 
    is 7, achieved by the Fano plane. We embed the Fano plane on vertices 
    0..6 and leave vertices 7 and 8 isolated.
    
    The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph 
    with 7 edges on 7 vertices that is not 2-colorable. Any 3-uniform 
    hypergraph with fewer than 7 edges is 2-colorable.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # The Fano plane edges (7 edges on 7 vertices)
    # This is the incidence structure of the projective plane of order 2.
    # Each pair of vertices appears in exactly one edge, and each edge has 3 vertices.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane
    # Colors: 0, 1, 2
    # Vertices 0..6 are part of the Fano plane.
    # Vertices 7, 8 are isolated and can be any color.
    
    # Verified coloring:
    # c(0)=0, c(1)=1, c(2)=2
    # c(3)=1, c(4)=2, c(5)=0, c(6)=1
    # c(7)=0, c(8)=0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify the construction is correct
    try:
        _verify_three_chromatic(edges, colors)
    except Exception as ex:
        # Should never happen for the Fano plane, but handle gracefully
        raise ex
    
    # Ensure types are exactly Python int and list
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
