<<<<<<< SEARCH
    # Collision penalty
    pen = 0.0
    margin = 0.0005
    for i in range(n):
        for j in range(i + 1, j + 1):
            pass
    for i in range(n):
        for j in range(i + 1, n):
            d = _edge_min_dist(verts[i], verts[j])
            if d < margin:
                pen += (margin - d) ** 2
    return side, pen
=======
    # Collision penalty
    pen = 0.0
    margin = 0.0005
    for i in range(n):
        for j in range(i + 1, n):
            d = _edge_min_dist(verts[i], verts[j])
            if d < margin:
                pen += (margin - d) ** 2
    return side, pen
>>>>>>> REPLACE
