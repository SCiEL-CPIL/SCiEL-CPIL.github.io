def construct():
    n = 16
    edges = []
    # Partition vertices into two sets of size 8
    A = range(8)
    B = range(8, 16)
    
    # Generate all edges between A and B
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges
