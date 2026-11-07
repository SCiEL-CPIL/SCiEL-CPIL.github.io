import time

def construct():
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices
    # with chromatic number 3 and minimum edge count (7).
    # Vertices 7 and 8 are isolated to satisfy the 9-vertex constraint.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0-6.
    # Colors must be in {0, 1, 2}.
    # Vertex 0: 0
    # Vertex 1: 1
    # Vertex 2: 2
    # Vertex 3: 0
    # Vertex 4: 2
    # Vertex 5: 1
    # Vertex 6: 0
    # Check edge (0,1,2): 0,1,2 (non-mono)
    # Check edge (0,3,4): 0,0,2 (non-mono)
    # Check edge (0,5,6): 0,1,0 (non-mono)
    # Check edge (1,3,5): 1,0,1 (non-mono)
    # Check edge (1,4,6): 1,2,0 (non-mono)
    # Check edge (2,3,6): 2,0,0 (non-mono)
    # Check edge (2,4,5): 2,2,1 (non-mono)
    
    colors = [0, 1, 2, 0, 2, 1, 0, 0, 0]
    
    # Ensure types are strictly Python int and list
    # Convert edges to list of lists of ints
    out_edges = []
    for e in edges:
        out_edges.append([int(e[0]), int(e[1]), int(e[2])])
        
    # Ensure colors are Python ints
    out_colors = [int(c) for c in colors]
    
    return {
        'edges': out_edges,
        'colors': out_colors
    }
