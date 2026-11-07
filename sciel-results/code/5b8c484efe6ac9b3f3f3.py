def construct():
    n = 16
    edges = []
    # Partition 0: 0-7
    # Partition 1: 8-15
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
