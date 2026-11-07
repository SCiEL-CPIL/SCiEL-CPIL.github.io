def construct():
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append([i, j])
    return edges
