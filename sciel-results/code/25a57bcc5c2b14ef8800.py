def _hex_init(n_points, seed=42):
    np.random.seed(seed)
    
    # Estimate spacing
    r_est = np.sqrt(0.9 / (n_points * np.pi))
    s = 2 * r_est
    h = s * np.sqrt(3) / 2
    
    # Generate a grid of points in a larger area, then crop/center
    # We need enough points to select n_points from.
    # Let's generate a 20x20 hex grid.
    
    points = []
    # y from -1 to 1
    y = -1.0
    row = 0
    while y <= 1.0:
        x = -1.0
        while x <= 1.0:
            # Hex shift
            offset = (s / 2) if (row % 2 == 1) else 0
            points.append((x + offset, y))
            x += s
        y += h
        row += 1
        
    points = np.array(points)
    
    # Filter points that are inside [-0.6, 0.6] to allow some room for movement
    mask = (points[:, 0] >= -0.6) & (points[:, 0] <= 0.6) & (points[:, 1] >= -0.6) & (points[:, 1] <= 0.6)
    points = points[mask]
    
    # If we have more than n_points, select the n_points closest to the center?
    # Or just the first n_points?
    # Sorting by distance to center and taking the closest n_points creates a circular shape.
    # This might not fill the square corners well.
    # However, the optimizer will push them to the corners.
    # Let's try sorting by distance to center.
    
    dist_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dist_to_center)
    points = points[indices[:n_points]]
    
    # Add jitter
    jitter_scale = s * 0.1
    points += np.random.normal(0, jitter_scale, points.shape)
    
    # Center the points
    points -= np.mean(points, axis=0)
    
    # Scale to fit?
    # The optimizer will handle scaling/positioning.
    # But if they are too far out, bounds might clip them immediately.
    # Let's ensure they are within [-0.5, 0.5] roughly.
    # If max coord > 0.5, scale down.
    max_c = np.max(np.abs(points))
    if max_c > 0.45:
        points *= 0.45 / max_c
        
    return points
