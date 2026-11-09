def construct():
    n = 32
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([4 * b + i, 4 * b + j])
    return {'n': n, 'edges': edges}
