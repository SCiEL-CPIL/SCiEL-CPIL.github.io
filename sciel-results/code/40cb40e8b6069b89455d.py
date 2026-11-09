def construct():
    n = 16
    # Partition vertices into two sets of size 8
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    # Connect every vertex in set_a to every vertex in set_b
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
