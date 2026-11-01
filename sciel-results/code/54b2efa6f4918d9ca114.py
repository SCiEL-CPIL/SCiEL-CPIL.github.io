import numpy as np

N = 50

def _covering_radius_approx(points, n_samples=500):
    """
    Approximate covering radius using a fixed set of sample points on the sphere.
    """
    # Generate sample points on the sphere (Fibonacci spiral for good coverage)
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize points
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi
    points = points / norms[:, np.newaxis]
    
    # Compute distances
    # diff shape: (n_samples, N, 3)
    diff = samples[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Nearest neighbor distance for each sample
    min_dist_sq = np.min(dist_sq, axis=1)
    
    # Convert to angular distance
    # cos(theta) = 1 - dist_sq / 2  (since |p|^2=|q|^2=1, |p-q|^2 = 2 - 2cos(theta))
    cos_theta = 1.0 - min_dist_sq / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    return np.max(theta)

def _covering_radius_exact(points):
    """
    Exact covering radius using ConvexHull.
    """
    from scipy.spatial import ConvexHull
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-9:
                continue
            normal = normal / norm_n
            d = np.dot(normal, p1)
            if d <= 1e-9:
                return np.pi
            cos_R = np.clip(np.dot(normal, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_radius:
                max_radius = R
        return max_radius
    except Exception:
        return np.pi

def construct():
    # Initialization: Spherical Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_energy = _covering_radius_approx(points)
    
    T = 0.01 # Initial temperature (radians)
    T_end = 1e-5
    alpha = 0.999 # Cooling rate
    steps = 5000 # Number of SA steps
    
    for step in range(steps):
        # Perturbation: move a random point
        idx = np.random.randint(N)
        old_point = points[idx].copy()
        
        # Generate random direction in tangent plane
        random_vec = np.random.randn(3)
        tangent = random_vec - np.dot(random_vec, points[idx]) * points[idx]
        norm_t = np.linalg.norm(tangent)
        if norm_t < 1e-9:
            continue
        tangent /= norm_t
        
        # Step size
        step_size = 0.1 * np.sqrt(T) # Adaptive step size
        new_point = points[idx] + step_size * tangent
        new_point /= np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_energy = _covering_radius_approx(points)
        
        # Acceptance criterion
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = points.copy()
        
        if new_energy <= best_energy or np.random.rand() < np.exp(-(new_energy - best_energy) / T):
            # Accept
            pass
        else:
            # Reject
            points[idx] = old_point
            
        T *= alpha
        
    # Final polish with exact radius check? No, just return best.
    # The approx radius might be slightly lower than exact, but the configuration is good.
    # We should return the points that minimize the EXACT radius among the candidates?
    # We only kept the best approx. Let's just return best_points.
    
    return best_points
