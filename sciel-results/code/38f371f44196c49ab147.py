import time

def construct():
    # Construct the Grötzsch graph (Mycielski of C5) on vertices 0..10
    # Vertices 0..4 form C5
    # Vertices 5..9 are copies of 0..4
    # Vertex 10 is the root
    
    n = 16
    edges = []
    
    # Original C5 edges
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        
    # Mycielski edges
    # For each edge (u, v) in C5, add (u, v') and (v, u')
    # v' is 5+v
    for i in range(5):
        u = i
        v = (i + 1) % 5
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
        
    # Root vertex 10 connected to all copies 5..9
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Current edges: 20
    # Vertices used: 0..10
    
    # Add vertices 11..15
    # The set of copy vertices 5..9 is an independent set in the Grötzsch graph.
    # We can connect new vertices to this independent set without creating triangles.
    # Connect each new vertex to all 5 copy vertices.
    
    for new_v in range(11, 16):
        for copy_v in range(5, 10):
            edges.append([new_v, copy_v])
            
    # Total edges: 20 + 5 * 5 = 45
    
    return edges
