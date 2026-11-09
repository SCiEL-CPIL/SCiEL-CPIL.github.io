import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Base graph G: 7 vertices, 10 edges, triangle-free, 3-chromatic
    # Vertices 0..6
    # Cycle: 0-1-2-3-4-5-6-0
    # Chords: 0-3, 1-4, 2-5
    base_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0],
        [0, 3], [1, 4], [2, 5]
    ]
    
    # Mycielski Construction
    # Vertices 0..6 -> G
    # Vertices 7..13 -> G' (7+u for u in 0..6)
    # Vertex 14 -> Top
    
    edges = [list(e) for e in base_edges]
    
    # Add edges from G' to G
    for u, v in base_edges:
        u_prime = u + 7
        v_prime = v + 7
        edges.append([u_prime, v])
        edges.append([v_prime, u])
        
    # Add edges from Top to G'
    for u in range(7):
        edges.append([u + 7, 14])
        
    # Add vertex 15 connected to G' (7..13)
    # G' vertices form an independent set in M(G)
    for u in range(7):
        edges.append([15, u + 7])
        
    return edges
