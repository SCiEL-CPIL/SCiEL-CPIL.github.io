"""
Constructs the minimal 3-uniform 3-chromatic hypergraph on 9 vertices.
The core structure is the Fano plane (7 vertices, 7 edges), which is the 
smallest 3-uniform hypergraph with chromatic number 3.
Vertices 7 and 8 are isolated.
"""
import time

def construct():
    """
    Returns a dict with keys 'edges' and 'colors'.
    'edges': List of 7 distinct triples on vertices 0..8.
    'colors': List of 9 integers in {0, 1, 2}.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Fano plane edges on vertices 0..6
    # The Fano plane lines are:
    # {0, 1, 6}, {0, 2, 5}, {0, 3, 4}, 
    # {1, 2, 4}, {1, 3, 5}, 
    # {2, 3, 6}, {4, 5, 6}
    
    edges_raw = [
        (0, 1, 6),
        (0, 2, 5),
        (0, 3, 4),
        (1, 2, 4),
        (1, 3, 5),
        (2, 3, 6),
        (4, 5, 6)
    ]
    
    # Convert to lists of Python ints
    edges = [list(e) for e in edges_raw]
    edges = [[int(x) for x in e] for e in edges]
    
    # Valid 3-coloring for Fano plane:
    # Color 0: vertices 0, 1, 2
    # Color 1: vertices 3, 4, 5
    # Color 2: vertex 6
    # Isolated vertices 7, 8 can be any color, e.g., 0.
    
    colors_raw = [0, 0, 0, 1, 1, 1, 2, 0, 0]
    colors = [int(c) for c in colors_raw]
    
    # Check deadline (should never trigger for this static construction)
    if time.time() > deadline:
        pass # Just in case, though it's instant
    
    return {
        'edges': edges,
        'colors': colors
    }
