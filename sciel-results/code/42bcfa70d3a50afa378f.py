def construct():
    edges = []
    for u in range(16):
        xu = u & 3
        yu = u >> 2
        for v in range(u + 1, 16):
            xv = v & 3
            yv = v >> 2
            if xu != xv and yu != yv:
                edges.append([u, v])
    return edges
