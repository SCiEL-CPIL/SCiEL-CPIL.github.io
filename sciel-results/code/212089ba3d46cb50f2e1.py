def construct():
    # Grötzsch graph edges
    # u: 0-4, v: 5-9, w: 10
    edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # v_i connected to u_{i-1} and u_{i+1}
    for i in range(5):
        edges.append([i, 5 + (i + 1) % 5])
        edges.append([i, 5 + (i - 1) % 5])
    # w (10) connected to all v_i
    for i in range(5):
        edges.append([10, 5 + i])
    
    # Add 5 new vertices 11-15
    # Connect each to all v_i (5-9)
    for new_v in range(11, 16):
        for v_i in range(5, 10):
            edges.append([new_v, v_i])
            
    return edges
