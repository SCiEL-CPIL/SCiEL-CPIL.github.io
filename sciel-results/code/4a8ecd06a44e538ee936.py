def construct():
    n = 16
    left = list(range(8))
    right = list(range(8, 16))
    edges = []
    for u in left:
        for v in right:
            edges.append([u, v])
    return edges
