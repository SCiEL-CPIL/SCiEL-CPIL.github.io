    # Inside the restart loop
    deg_counts = [0] * 16
    current_edges = set()
    # ...
    for u, v in shuffled_edges:
        if deg_counts[u] >= 3 or deg_counts[v] >= 3:
            continue
        if not adds_q3(current_edges, u, v, vertices, deg_counts):
            current_edges.add((u, v))
            deg_counts[u] += 1
            deg_counts[v] += 1
