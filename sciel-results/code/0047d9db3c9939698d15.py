import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Vertices 0..19 in A, 20..39 in B
    # Base K20,20
    # A = 0..19, B = 20..39
    
    # We want to add a star of size k=10 in A
    # Center: 0
    # Leaves: 1..10
    # Internal edges: (0, i) for i in 1..10
    
    # Remove k-1 = 9 edges from B incident to center 0
    # Remove (0, 20), (0, 21), ..., (0, 28)
    
    edges = []
    
    # Add all bipartite edges
    for u in range(20):
        for v in range(20, 40):
            # Skip removed edges
            if u == 0 and 20 <= v <= 28:
                continue
            edges.append([u, v])
            
    # Add internal star edges
    for i in range(1, 11):
        edges.append([0, i])
        
    # Check count
    # Should be 401
    
    return edges
