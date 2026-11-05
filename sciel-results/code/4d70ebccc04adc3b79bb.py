def _build_seed_graph():
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges.append([a,b])
        edges.append([a,c])
        edges.append([b,c])
    return 30, edges
