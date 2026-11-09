def construct():
    n = 16
    half = n // 2
    # Construct K_{8,8}
    # Vertices 0..7 in left partition, 8..15 in right partition
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges
