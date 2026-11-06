def construct():
    # Number-theoretic construction:
    # Select 30 points as a linear combination of two primitive lattice vectors u and v.
    # u = (1, 0), v = (0, 1).
    # The set is { i*u + j*v | i in [0, 4], j in [0, 5] }.
    # This forms a 5x6 rectangular grid.
    # We shift the grid to be centered within [-30, 30]^2.
    
    W = 5
    H = 6
    
    points = []
    for i in range(W):
        for j in range(H):
            # Center the grid:
            # i ranges 0..4, center is 2.
            # j ranges 0..5, center is 2.5.
            # To keep integers, we can shift by -2 for x, -2 for y.
            x = i - 2
            y = j - 2
            points.append((x, y))
            
    # Verify constraints
    assert len(set(points)) == 30
    assert all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in points)
    
    return points
