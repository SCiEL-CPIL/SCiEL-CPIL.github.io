import time

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices that is not 2-colorable.
    # It is known to be 3-chromatic.
    # We extend it to 9 vertices by adding two isolated vertices (7 and 8).
    # The edge count is 7.
    
    # Edges of the Fano plane (indices 0-6)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane extended to 9 vertices.
    # Vertices 0-6 are colored such that no edge is monochromatic.
    # Vertices 7 and 8 are isolated, so their color doesn't affect edge validity, 
    # but they must be from 0..2.
    # Coloring:
    # 0: 0
    # 1: 1
    # 2: 1  (Edge 0,1,2: 0,1,1 - OK)
    # 3: 2
    # 4: 2  (Edge 0,3,4: 0,2,2 - OK)
    # 5: 1
    # 6: 2  (Edge 0,5,6: 0,1,2 - OK)
    # Check remaining edges:
    # 1,3,5: 1,2,1 - OK
    # 1,4,6: 1,2,2 - OK
    # 2,3,6: 1,2,2 - OK
    # 2,4,5: 1,2,1 - OK
    
    colors = [0, 1, 1, 2, 2, 1, 2, 0, 0]
    
    # Ensure types are exactly Python int and list
    edges_out = [list(e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {
        'edges': edges_out,
        'colors': colors_out
    }
