def construct():
    n = 16
    half = n // 2
    A = list(range(half))
    B = list(range(half, n))
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    return edges
