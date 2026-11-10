def construct():
    n = 16
    mid = n // 2
    partition_a = list(range(mid))
    partition_b = list(range(mid, n))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    return edges
