"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges that is 3-chromatic.
    # Any 3-uniform hypergraph with <= 6 edges is 2-colorable.
    # Thus, 7 is the global minimum for the number of edges.
    # We define the Fano plane on vertices 0..6 and leave 7, 8 isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 1
    # 5: 2
    # 6: 2
    # 7: 0 (isolated)
    # 8: 0 (isolated)
    
    colors = [0, 1, 2, 1, 1, 2, 2, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
