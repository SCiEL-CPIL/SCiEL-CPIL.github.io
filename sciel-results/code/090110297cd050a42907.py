import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_circumradii(points, hull):
    """
    Compute circumradii and circumcenters for all hull facets.
    Returns:
        radii: (M,) array of circumradii (angular)
        centers: (M, 3) array of circumcenters on sphere
        facets: (M, 3) array of facet indices
    """
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    cross_sq = np.einsum('ij,ij->i', cross, cross)
    
    valid = cross_sq > 1e-28
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        # Circumcenter formula for triangle on unit sphere
        # The circumcenter in 3D space is p1 + t * cross where t = (cross . p1) / (cross . cross)
        # This gives the center of the circumscribed circle in 3D, which lies on the line
        # through the origin and the normalized cross product direction.
        # For points on the unit sphere, the circumcenter on the sphere is simply
        # the normalized cross product (or its negative).
        # Let's use the normalized cross product approach which is more stable.
        c_raw = cross[valid]
        norms = np.linalg.norm(c_raw, axis=1, keepdims=True)
        c = c_raw / np.maximum(norms, 1e-15)
        
        # Ensure centers point to the empty cap side (away from origin, toward exterior)
        centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dots = np.einsum('ij,ij->i', c, centroids)
        flip = dots < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        
        # Compute angular distances from circumcenter to each vertex
        dot_vals = np.einsum('ij,ij->i', c, p1[valid])
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        radii[valid] = np.arccos(dot_vals)
    
    return radii, centers, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        radii, centers, facets = _compute_circumradii(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def _gradient_balanced(points, hull, temperature=0.5):
    """
    Compute a gradient that balances circumradii across all facets.
    Uses a soft-max weighting to prioritize larger facets but still
    contributes to all facets, enabling escape from local minima.
    
    Returns:
        gradient: (N, 3) array of gradient directions
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    
    # Soft-max weighting: weight_k = exp(r_k / T) / sum(exp(r_j / T))
    # This gives more weight to larger facets but non-zero weight to all
    if temperature < 1e-10:
        # Hard max: only the largest facet
        max_idx = np.argmax(radii)
        weights = np.zeros(len(radii))
        weights[max_idx] = 1.0
    else:
        # Numerically stable soft-max
        shifted = radii - np.max(radii)
        exp_vals = np.exp(shifted / temperature)
        weights = exp_vals / np.sum(exp_vals)
    
    gradient = np.zeros_like(points)
    
    for k in range(len(facets)):
        if weights[k] < 1e-10:
            continue
            
        c = centers[k]
        tri = facets[k]
        w = weights[k]
        
        for idx in tri:
            p = points[idx]
            # Direction from p to c
            v = c - p
            # Project onto tangent plane at p
            dot_vp = np.dot(v, p)
            tangent_v = v - dot_vp * p
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-14:
                gradient[idx] += w * tangent_v / norm_tv
    
    return gradient

def _gradient_max_only(points, hull, threshold=0.001):
    """
    Compute gradient focusing only on facets near the maximum.
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    max_r = np.max(radii)
    
    threshold_val = max_r - threshold
    critical = radii > threshold_val
    
    if not np.any(critical):
        return np.zeros_like(points)
    
    gradient = np.zeros_like(points)
    
    for k in np.where(critical)[0]:
        c = centers[k]
        tri = facets[k]
        weight = (radii[k] - threshold_val) / (max_r - threshold_val + 1e-10)
        
        for idx in tri:
            p = points[idx]
            v = c - p
            dot_vp = np.dot(v, p)
            tangent_v = v - dot_vp * p
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-14:
                gradient[idx] += weight * tangent_v / norm_tv
    
    return gradient

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
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    min_step_size = 1e-8
    no_improve_count = 0
    max_no_improve = 30
    
    # Phase 1: Balanced gradient with soft-max to explore broadly
    # Phase 2: Transition to max-only for fine-tuning
    # Phase 3: Very fine max-only tuning
    
    # Phase 1: Balanced (0-60% of time)
    phase1_end = start_time + (deadline - start_time) * 0.60
    step_size = 0.02
    temperature = 0.3
    
    while time.time() < phase1_end:
        try:
            hull = ConvexHull(points)
        except Exception:
            kick = rng.normal(0, 0.02, points.shape)
            points = points + kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
        
        grad = _gradient_balanced(points, hull, temperature)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            step_size = min(step_size * 1.02, 0.05)
            # Decrease temperature over time to focus more
            temperature = max(temperature * 0.995, 0.05)
        else:
            step_size = max(step_size * 0.93, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                kick = rng.normal(0, 0.03, points.shape)
                points = points + kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.01
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    # Phase 2: Max-only gradient with moderate threshold (60-85% of time)
    phase2_end = start_time + (deadline - start_time) * 0.85
    step_size = 0.005
    threshold = 0.002
    no_improve_count = 0
    
    while time.time() < phase2_end:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        grad = _gradient_max_only(points, hull, threshold)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            step_size = min(step_size * 1.01, 0.01)
        else:
            step_size = max(step_size * 0.95, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > 40:
                kick = rng.normal(0, 0.01, points.shape)
                points = points + kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.002
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    # Phase 3: Very fine max-only tuning (85-100% of time)
    step_size = 0.001
    threshold = 0.0005
    no_improve_count = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        grad = _gradient_max_only(points, hull, threshold)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
        else:
            step_size = max(step_size * 0.95, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > 60:
                break
    
    # Ensure output is valid
    result = best_points.copy()
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    
    # Final validity check
    final_radius = _compute_covering_radius(result)
    if final_radius >= 180.0:
        # Fallback: return Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        result = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        result /= np.linalg.norm(result, axis=1, keepdims=True)
    
    return result
