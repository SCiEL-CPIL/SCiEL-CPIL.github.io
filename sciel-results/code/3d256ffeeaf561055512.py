import time

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n = 16
    # K_3,13 with all edges in the part of size 3
    # Vertices 0,1,2 form the clique K_3
    # Vertices 3..15 form the independent set
    edges = []
    
    # Edges between 0..2 and 3..15
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
            
    # Edges within 0..2
    edges.append([0, 1])
    edges.append([0, 2])
    edges.append([1, 2])
    
    # Check validity (optional, but good for debugging)
    # This graph is known to be Q3-free.
    # Return the list of edges
    return edges
