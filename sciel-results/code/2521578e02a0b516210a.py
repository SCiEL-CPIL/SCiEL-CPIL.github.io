def construct():
    # 8 disjoint K4s plus one universal-ish vertex connected to exactly 1 vertex per K4.
    # This is C5-free and has independence number 8.
    n = 33
    edges = []
    # K4 blocks
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * b + i
                v = 4 * b + j
                edges.append([u, v])
    # Connect vertex 32 to one vertex in each K4 (vertex 4*b)
    for b in range(8):
        edges.append([32, 4 * b])
    return {'n': n, 'edges': edges}
