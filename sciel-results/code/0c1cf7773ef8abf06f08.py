import numpy as np

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_polygon(epsilon):
        # Radial modulation with frequency 2 (ellipse-like)
        radii = 1.0 + epsilon * np.cos(2.0 * angles)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    def calculate_score(polygon):
        # Normalize
        centroid = np.mean(polygon, axis=0)
        p_norm = polygon - centroid
        dists_c = np.linalg.norm(p_norm, axis=1)
        mean_dist = np.mean(dists_c)
        if mean_dist <= 0.01:
            return 0.0
        p_norm = p_norm / mean_dist
        
        # Check convexity and repeated points
        # Compute cross products of consecutive edges
        edges = np.roll(p_norm, -1, axis=0) - p_norm
        cross = np.cross(edges, np.roll(edges, -1, axis=0))
        # Strictly convex: all cross products have the same sign and are non-zero
        if np.any(cross == 0) or not (np.all(cross > 0) or np.all(cross < 0)):
            return 0.0
            
        min_score = 1.0
        for i in range(n):
            v = p_norm[i]
            # Distances to all other vertices
            dists = np.linalg.norm(p_norm - v, axis=1)
            dists[i] = np.inf # Exclude self
            
            # Sort distances
            sorted_dists = np.sort(dists)
            d = sorted_dists[:19]
            
            # Check min distance
            if d[0] < 0.05:
                return 0.0
            
            # Find k
            gaps = np.diff(d)
            k = np.argmin(gaps)
            
            D = (d[k] + d[k+1]) / 2.0
            if D < 0.1:
                return 0.0
            
            # Select 4 distances
            if k == 0:
                four_d = d[0:4]
            elif k == 17:
                four_d = d[15:19]
            else:
                four_d = d[k-1:k+3]
                
            ratios = np.maximum(four_d / D, D / four_d)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)
            
            if score < min_score:
                min_score = score
                
        return min_score

    # Optimize epsilon
    best_eps = 0.0
    best_score = 0.0
    
    # Search for epsilon in a small range
    # The seed used angle perturbation. Radial perturbation might help.
    for eps in np.linspace(-0.2, 0.2, 41):
        p = get_polygon(eps)
        s = calculate_score(p)
        if s > best_score:
            best_score = s
            best_eps = eps
            
    # If the best score is low, try frequency 4 or 10?
    # For now, stick to freq 2.
    
    return get_polygon(best_eps)
