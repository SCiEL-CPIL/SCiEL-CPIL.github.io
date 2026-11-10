def construct():
    n = 15
    # K_{7,8}: part A = {0..6}, part B = {7..14}
    edges = []
    for i in range(7):
        for j in range(7, 15):
            edges.append([i, j])
    return edges
