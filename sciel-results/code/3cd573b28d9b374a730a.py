def construct():
    points = []
    # Generate points in a diamond shape (Manhattan distance <= 4)
    for y in range(-4, 5):
        for x in range(-4, 5):
            if abs(x) + abs(y) <= 4:
                points.append((x, y))
    
    # We have 41 points. We need 30.
    # Sort by Manhattan distance to keep the most central points
    points.sort(key=lambda p: abs(p[0]) + abs(p[1]))
    
    # Take the first 30
    return points[:30]
