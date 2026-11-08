def objective(x, beta):
    points = x.reshape(N, 2)
    
    # Calculate hull area to normalize
    hull_area = convex_hull_area(points)
    if hull_area < 1e-6:
        return 1e6
        
    # Scale points to unit hull area
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # Now hull_area is 1.0
    # Calculate convexity penalty on scaled points
    pen = convexity_penalty(points)
    
    # Calculate triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                areas.append(triangle_area(points[i], points[j], points[k]))
    areas = np.array(areas)
    
    # Soft min
    min_area_approx = soft_min(areas, beta)
    
    # Objective: Maximize min area, so minimize -min_area
    # Add penalty for non-convexity
    return -min_area_approx + 10.0 * pen
