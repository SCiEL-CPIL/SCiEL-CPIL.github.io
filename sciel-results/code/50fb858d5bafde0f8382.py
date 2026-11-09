<<<<<<< SEARCH
def _generate_hex_init():
    """
    Generate an initial hexagonal packing.
    """
    # Target N=200
    # Hex grid spacing s
    # Area per point in hex grid is s^2 * sqrt(3)/2
    # Total Area = 1.0
    # N * s^2 * sqrt(3)/2 = 1.0  => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points in a grid larger than the square
    # Square is [-0.5, 0.5]
    # We need to cover this area.
    
    points = []
    # y range
    dy = s * np.sqrt(3.0) / 2.0
    # Start from -0.6 to 0.6 to ensure coverage
    y = -0.6
    row = 0
    while y <= 0.6:
        # x offset for alternating rows
        offset = 0.0 if row % 2 == 0 else s / 2.0
        x = -0.6 + offset
        while x <= 0.6:
            points.append((x, y))
            x += s
        y += dy
        row += 1
    
    points = np.array(points)
    
    # Filter points inside the square [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, remove the ones with the worst local packing
    if len(points) > N:
        # Calculate min distance to neighbor for each point
        # This is O(M^2) where M is number of points. M is approx 200-300.
        # 300^2 = 90,000. Fast.
        min_dists = np.full(len(points), np.inf)
        for i in range(len(points)):
            dists = np.linalg.norm(points[i] - points, axis=1)
            dists[i] = np.inf
            min_dists[i] = np.min(dists)
        
        # Also consider wall distance?
        # Let's just use min neighbor distance for now.
        # Sort indices by min_dist, remove the smallest ones
        idx = np.argsort(min_dists)
        # Keep the largest N
        keep_idx = idx[N:]
        points = points[keep_idx]
        
    # If we have fewer than N points (unlikely with this s), we might need to add
    # But for N=200, s is small enough that we usually have >200 points in the square.
    # If len(points) < N, we just pad with random points? Or just return what we have?
    # The task requires exactly N.
    if len(points) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    # Jitter slightly to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    return points
=======
def _generate_hex_init():
    """
    Generate an initial hexagonal packing.
    """
    # Target N=200
    # Hex grid spacing s
    # Area per point in hex grid is s^2 * sqrt(3)/2
    # Total Area = 1.0
    # N * s^2 * sqrt(3)/2 = 1.0  => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points in a grid larger than the square
    # Square is [-0.5, 0.5]
    # We need to cover this area.
    
    points = []
    # y range
    dy = s * np.sqrt(3.0) / 2.0
    # Start from -0.6 to 0.6 to ensure coverage
    y = -0.6
    row = 0
    while y <= 0.6:
        # x offset for alternating rows
        offset = 0.0 if row % 2 == 0 else s / 2.0
        x = -0.6 + offset
        while x <= 0.6:
            points.append((x, y))
            x += s
        y += dy
        row += 1
    
    points = np.array(points)
    
    # Filter points inside the square [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, remove the ones with the worst local packing
    if len(points) > N:
        # Calculate min distance to neighbor for each point
        min_dists = np.full(len(points), np.inf)
        for i in range(len(points)):
            dists = np.linalg.norm(points[i] - points, axis=1)
            dists[i] = np.inf
            min_dists[i] = np.min(dists)
        
        idx = np.argsort(min_dists)
        keep_idx = idx[N:]
        points = points[keep_idx]
        
    if len(points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    # Jitter slightly to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    return points

def _repulsive_relaxation(centers, n_iters=50, step_scale=0.5, deadline=None):
    """
    Iteratively push circles apart from their nearest neighbors and from walls.
    This is a simple repulsion-based relaxation that helps escape local minima
    in the hex grid initialization.
    """
    centers = centers.copy()
    start_time = time.time()
    
    for it in range(n_iters):
        if deadline is not None and time.time() - start_time > deadline:
            break
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Force on each circle from neighbors
        # For each pair i,j: force on i is (c_i - c_j) / d_ij^2 * strength
        # Only apply if d_ij < threshold (to limit interactions)
        threshold = 0.1  # Only interact with circles within this distance
        
        forces = np.zeros_like(centers)
        for i in range(N):
            # Find neighbors within threshold
            mask = dists[i] < threshold
            mask[i] = False  # Exclude self
            if not np.any(mask):
                continue
            
            neighbors = centers[mask]
            d = dists[i][mask]
            # Force direction: away from each neighbor
            # Strength: 1/d^2 to make it strong when close
            dirs = (centers[i] - neighbors) / d[:, np.newaxis]
            strengths = 1.0 / (d**2 + 1e-6)
            forces[i] = np.sum(dirs * strengths[:, np.newaxis], axis=0)
        
        # Wall repulsion: push away from walls
        for i in range(N):
            for k in range(2):
                if centers[i, k] < -0.5:
                    forces[i, k] += 10.0
                elif centers[i, k] > 0.5:
                    forces[i, k] -= 10.0
                # Soft wall repulsion
                dist_to_wall = 0.5 - np.abs(centers[i, k])
                if dist_to_wall < 0.05:
                    repel_strength = (0.05 - dist_to_wall) / 0.05 * 5.0
                    if centers[i, k] < 0:
                        forces[i, k] += repel_strength
                    else:
                        forces[i, k] -= repel_strength
        
        # Apply forces with adaptive step size
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_mags) if np.any(force_mags > 0) else 1.0
        step = step_scale / (max_force + 1e-6)
        
        centers += step * forces
        centers = np.clip(centers, -0.499, 0.499)
        
        # Decay step size
        step_scale *= 0.95
    
    return centers
>>>>>>> REPLACE

