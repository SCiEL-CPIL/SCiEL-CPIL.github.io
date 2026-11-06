def construct():
    points = []
    for a in range(6):
        for b in range(6):
            # Exclude corners
            if (a == 0 and b == 0) or (a == 5 and b == 0) or (a == 0 and b == 5) or (a == 5 and b == 5):
                continue
            points.append([a, b])
    return points
