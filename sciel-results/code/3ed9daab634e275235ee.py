def construct():
    n = 16
    h = n // 2
    edges = [[u, v] for u in range(h) for v in range(h, n)]
    return edges
