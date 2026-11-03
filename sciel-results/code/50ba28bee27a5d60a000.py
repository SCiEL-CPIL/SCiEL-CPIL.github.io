import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        M = len(facets)
        
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross, axis=1)
        
        valid = norm_cross > 1e-14
        if not np.any(valid):
            return 180.0
            
        c = np.zeros((M, 3))
        c[valid] = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1 + p2 + p3) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        dot_val = np.clip(np.sum(c * p1, axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        dist = np.where(dist > np.pi / 2, np.pi - dist, dist)
        
        return np.degrees(np.max(dist))
        
    except Exception:
        return 180.0

def _get_facet_data(points):
    """
    Compute circumcenters and radii for all hull facets (vectorized).
    Returns centers (M, 3), radii (M,), and facets (M, 3).
    """
    hull = ConvexHull(points)
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = np.zeros((M, 3))
        c[valid] = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1 + p2 + p3) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers = c
        
        dot_val = np.clip(np.sum(c * p1, axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        dist = np.where(dist > np.pi / 2, np.pi - dist, dist)
        radii = dist
    
    return centers, radii, facets

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Simulated annealing parameters
    T = 0.1  # Initial temperature (in radians)
    T_min = 1e-6
    cooling_rate = 0.995
    
    current_points = points.copy()
    current_radius = best_radius
    T_current = T
    
    while time.time() < deadline:
        # Get facet data
        try:
            centers, radii, facets = _get_facet_data(current_points)
        except Exception:
            # If hull fails, accept a random kick
            kick = rng.normal(0, T_current, current_points.shape)
            current_points = current_points + kick
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            continue
        
        # For each point, find the adjacent triangle with the maximum radius
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        for k in range(len(facets)):
            r_k = radii[k]
            if r_k <= 0:
                continue
            c_k = centers[k]
            tri = facets[k]
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        # Generate a proposal: move points towards their max center with noise
        move = np.zeros_like(current_points)
        for i in range(N):
            if point_max_radius[i] > 0:
                v = point_max_center[i] - current_points[i]
                dot_vp = np.dot(v, current_points[i])
                tangent_v = v - dot_vp * current_points[i]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    move[i] = tangent_v / norm_tv
                    move[i] *= point_max_radius[i]
        
        # Add noise scaled by temperature
        noise = rng.normal(0, T_current, current_points.shape)
        proposed = current_points + 0.1 * move + noise
        proposed /= np.linalg.norm(proposed, axis=1, keepdims=True)
        
        proposed_radius = _compute_covering_radius(proposed)
        
        # Metropolis acceptance criterion
        delta = proposed_radius - current_radius
        if delta < 0 or (T_current > T_min and rng.random() < np.exp(-delta / T_current)):
            current_points = proposed
            current_radius = proposed_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            # Rejected, but still cool down
            pass
        
        # Cool down
        T_current *= cooling_rate
        if T_current < T_min:
            T_current = T_min
    
    return best_points
