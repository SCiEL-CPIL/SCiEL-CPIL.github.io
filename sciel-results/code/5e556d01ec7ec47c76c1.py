    def construct():
        n = 30
        edges = []
        # Vertices are (i, j) where i in 0..4, j in 0..5
        # Index v = i * 6 + j
        for i in range(5):
            for j in range(6):
                v1 = i * 6 + j
                for i2 in range(5):
                    for j2 in range(6):
                        v2 = i2 * 6 + j2
                        if v1 < v2:
                            if i == i2 or j == j2:
                                edges.append([v1, v2])
        return {'n': n, 'edges': edges}
    