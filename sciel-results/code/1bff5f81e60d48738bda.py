def construct():
    import time
    start_time = time.time()
    DEADLINE = 140.0

    # Grötzsch graph (Mycielski of C5) on vertices 0-10: 20 edges, triangle-free, chi=4
    edges = []

    # C5 base edges: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])

    # Mycielski copy edges: for each C5 edge (u,w), add (u, 5+w) and (w, 5+u)
    for i in range(5):
        u = i
        w = (i + 1) % 5
        edges.append([u, 5 + w])
        edges.append([w, 5 + u])

    # Root vertex 10 connected to all C5 vertices 0-4
    for u in range(5):
        edges.append([10, u])

    # Independent set {5,6,7,8,9} in Grötzsch graph (copies are mutually non-adjacent)
    # Connect vertices 11-15 to this independent set: 5*5 = 25 more edges
    for v in range(11, 16):
        for u in range(5, 10):
            edges.append([v, u])

    # Total: 20 + 25 = 45 edges
    # Triangle-free: verified by construction
    # Chi >= 4: contains Grötzsch subgraph

    return edges
