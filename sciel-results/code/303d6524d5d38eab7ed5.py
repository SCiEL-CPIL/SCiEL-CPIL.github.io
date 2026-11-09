def construct():
    pts = []
    for i in range(4):
        for j in range(8):
            pts.append((i, j))
    # Remove 2 points to get 30
    # Remove (0,0) and (3,7)
    pts.remove((0,0))
    pts.remove((3,7))
    return pts
