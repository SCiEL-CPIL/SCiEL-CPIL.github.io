    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[(i, j)] = idx
            edge_map[(j, i)] = idx
            idx += 1
