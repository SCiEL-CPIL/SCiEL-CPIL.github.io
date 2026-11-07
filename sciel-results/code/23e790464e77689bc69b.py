"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Return a 3-chromatic 3-uniform hypergraph on vertices 0..8 with exactly 7 edges,
    which is the theoretical minimum for a 3-chromatic 3-uniform hypergraph.
    
    The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph with 7
    vertices and 7 edges that is 3-chromatic. We embed it on vertices 0..6 and
    leave vertices 7 and 8 isolated.
    
    A valid 3-coloring is provided. The hypergraph is not 2-colorable.
    """
    start_time = time.time()
    deadline = start_time + 59.0 # Safety margin for 60s limit

    # Fano plane edges (7 triples on vertices 0..6)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring: each edge has at least two different colors
    # Vertex:  0  1  2  3  4  5  6  7  8
    # Color:   0  1  2  1  2  0  1  0  0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strictly Python int and list
    edges = [[int(x) for x in e] for e in edges]
    colors = [int(c) for c in colors]
    
    # Deadline guard (though not strictly necessary for O(1) logic)
    if time.time() > deadline:
        # Return best result found (which is the constant solution)
        pass
        
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
