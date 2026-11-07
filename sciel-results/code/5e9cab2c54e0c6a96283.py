def construct():
    n = 16
    # Partition vertices into two sets of 8
    set_a = list(range(0, 8))
    set_b = list(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
    
    return edges
