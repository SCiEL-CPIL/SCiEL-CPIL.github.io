def construct():
    edges = []
    # Component 1: K_{3,3} on 0-5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
    # Component 2: K_{3,3} on 6-11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
    return edges
