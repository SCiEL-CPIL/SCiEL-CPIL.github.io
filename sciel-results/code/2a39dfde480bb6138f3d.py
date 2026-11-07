"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the minimal 3-chromatic 3-uniform hypergraph (7 edges).
    # Any 3-uniform hypergraph with < 7 edges is 2-colorable.
    # We use vertices 0..6 for the Fano plane. Vertices 7, 8 are isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]

    # Valid 3-coloring for the Fano plane:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Vertices 7, 8 are isolated, assign color 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]

    # Enforce strict types: Python int and list
    edges_out = [ [int(x) for x in e] for e in edges ]
    colors_out = [int(c) for c in colors]

    return {'edges': edges_out, 'colors': colors_out}
# EVOLVE-BLOCK-END
