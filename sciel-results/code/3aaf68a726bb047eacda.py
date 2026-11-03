def generate_hex_init(R, C):
    points = []
    for r in range(R):
        for c in range(C):
            x = c + (0.5 if r % 2 else 0)
            y = r * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Uniform scale to fit
    scale = 1.0 / max(width, height)
    
    points = (points - (min_x + max_x)/2) * scale
    points[:, 1] = (points[:, 1] - (min_y + max_y)/2) * scale
    
    # Select N points closest to center
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)[:N]
    points = points[indices]
    
    return points
