def construct():
        edges = []
        # First K3,3 on 0..5
        for i in range(3):
            for j in range(3, 6):
                edges.append([i, j])
        # Second K3,3 on 6..11
        for i in range(6, 9):
            for j in range(9, 12):
                edges.append([i, j])
        return edges
