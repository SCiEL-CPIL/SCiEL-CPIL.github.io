def construct():
        n = 16
        edges = []
        for u in range(0, n, 2):
            for v in range(1, n, 2):
                edges.append([u, v])
        return edges
