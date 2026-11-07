def construct():
    N = 15
    edges = []
    for i in range(7):
        for j in range(7, N):
            edges.append([i, j])
    return edges
