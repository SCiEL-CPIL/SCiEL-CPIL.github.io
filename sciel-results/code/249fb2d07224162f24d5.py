def _concentric_rings(n, seed=0):
    """Generate points in concentric rings."""
    rng = np.random.RandomState(seed)
    points = []
    # Center point
    points.append([0, 0])
    remaining = n - 1
    rings = 2
    for r in range(1, rings + 1):
        if r == rings:
            pts_in_ring = remaining
        else:
            pts_in_ring = int(np.ceil(remaining / (rings - r + 1)))
            remaining -= pts_in_ring
        
        radius = r * 0.4
        angles = np.linspace(0, 2 * np.pi, pts_in_ring, endpoint=False)
        angles += rng.uniform(-0.1, 0.1, pts_in_ring)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points.extend(zip(x, y))
    
    return np.array(points[:n])

def _optimize_worst_triangles(points, time_limit, rng_seed=42):
    """
    Optimizes points by iteratively moving the vertices of the minimum area triangle
    to increase its area.
    """
    rng = np.random.RandomState(rng_seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() < time_limit:
        # Find the minimum area triangle
        min_area = float('inf')
        min_idx = None
        n = len(current)
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = current[i]
                    p2 = current[j]
                    p3 = current[k]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                     (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        
        if min_idx is None:
            break
            
        i, j, k = min_idx
        p1, p2, p3 = current[i], current[j], current[k]
        
        centroid = (p1 + p2 + p3) / 3.0
        
        improved = False
        
        for _ in range(10):
            new_points = current.copy()
            
            dir_i = p1 - centroid
            norm_i = np.linalg.norm(dir_i)
            if norm_i > 1e-9:
                dir_i /= norm_i
            
            dir_j = p2 - centroid
            norm_j = np.linalg.norm(dir_j)
            if norm_j > 1e-9:
                dir_j /= norm_j
                
            dir_k = p3 - centroid
            norm_k = np.linalg.norm(dir_k)
            if norm_k > 1e-9:
                dir_k /= norm_k
            
            noise_i = rng.normal(0, step_size, 2)
            noise_j = rng.normal(0, step_size, 2)
            noise_k = rng.normal(0, step_size, 2)
            
            new_points[i] = p1 + dir_i * step_size + noise_i
            new_points[j] = p2 + dir_j * step_size + noise_j
            new_points[k] = p3 + dir_k * step_size + noise_k
            
            new_score = _compute_score(new_points)
            
            if new_score > current_score:
                current = new_points
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.8
            if step_size < 1e-6:
                break
            if no_improve_count > 100:
                break
                
    return best, best_score
