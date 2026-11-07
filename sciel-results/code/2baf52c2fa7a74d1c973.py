"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return a 3-uniform hypergraph on 9 vertices (0..8) that is 3-chromatic
    (not 2-colorable) with the minimum possible number of edges for the
    specified construction (Fano plane + connecting edge).
    
    The Fano plane (7 edges) is not 2-colorable.
    We add one edge {0, 7, 8} to connect vertices 7 and 8 to the Fano plane,
    resulting in a connected 3-uniform hypergraph on 9 vertices.
    Total edges: 8.
    """
    # Fano plane edges on vertices 0-6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Add edge to include vertices 7 and 8
    edges.append([0, 7, 8])
    
    # Verify no 2-coloring exists
    # Brute force over 2^9 = 512 assignments
    from itertools import product
    for coloring in product([0, 1], repeat=9):
        valid = True
        for edge in edges:
            if len(set(coloring[v] for v in edge)) == 1:
                valid = False
                break
        if valid:
            raise ValueError("Found a 2-coloring, hypergraph should not be 2-colorable")
    
    # Define a valid 3-coloring
    # Fano plane coloring:
    # 0:0, 1:1, 2:2, 3:1, 4:0, 5:2, 6:1
    # Check Fano edges:
    # {0,1,2}: 0,1,2 (ok)
    # {0,3,4}: 0,1,0 (ok)
    # {0,5,6}: 0,2,1 (ok)
    # {1,3,5}: 1,1,2 (ok)
    # {1,4,6}: 1,0,1 (ok)
    # {2,3,6}: 2,1,1 (ok)
    # {2,4,5}: 2,0,2 (ok)
    
    # New edge {0, 7, 8}
    # colors[0] = 0.
    # Need colors[7], colors[8] such that {0, c7, c8} is not monochromatic.
    # Let c7=1, c8=2. {0,1,2} is ok.
    
    colors = [0, 1, 2, 1, 0, 2, 1, 1, 2]
    
    # Verify the provided coloring is valid
    for edge in edges:
        if len(set(colors[v] for v in edge)) == 1:
            raise ValueError("Provided coloring is not valid")
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
