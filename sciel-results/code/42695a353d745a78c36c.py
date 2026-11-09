"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
P = {'n': 9, 'r': 3, 'colors': 3}

def construct():
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges and chromatic number 3.
    # It is the minimum edge count for any 3-uniform hypergraph with chromatic number 3.
    # Vertices 7 and 8 are isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:1, 6:0
    # 7:0, 8:0
    colors = [0, 1, 2, 0, 1, 1, 0, 0, 0]
    
    # Enforce strict types
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
