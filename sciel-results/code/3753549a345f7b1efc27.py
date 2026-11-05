def construct():
    # Generate radius-3 hexagon points
    R = 3
    points = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a+b) <= R:
                points.append((a, b))
    
    # Define the 5 corners to remove to maximize edges (minimize degree loss)
    # These are 5 of the 6 corners of the hexagon
    remove_set = {
        (3, 0),
        (0, 3),
        (-3, 3),
        (-3, 0),
        (0, -3)
    }
    
    result = [list(p) for p in points if p not in remove_set]
    
    # Verification
    assert len(result) == 32
    return result
