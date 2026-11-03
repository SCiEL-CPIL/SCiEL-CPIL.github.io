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
        # Circumcenter via: p1 + t * cross, where t = (cross . p1) / (cross . cross)
        # This gives the 3D circumcenter; normalize to get spherical circumcenter
        dot_cp = np.einsum('ij,ij->i', cross[valid], p1[valid])
        t = dot_cp / cross_sq[valid]
        c_3d = p1[valid] + t[:, np.newaxis] * cross[valid]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(c_3d, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        c = c_3d / norms
        
        # Ensure centers point to the empty cap side (same side as centroid)
        centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dots = np.einsum('ij,ij->i', c, centroids)
        flip = dots < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        
        # Compute angular distances
        dot_vals = np.einsum('ij,ij->i', c, p1[valid])
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        radii[valid] = np.arccos(dot_vals)
    
    return radii, centers, facets

def _compute_covering_radius(points, hull=None):
    """
    Compute the covering radius in degrees.
    """
    try:
        if hull is None:
            hull = ConvexHull(points)
        radii, centers, facets = _compute_circumradii(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def _soft_max_gradient(points, hull, temperature=0.01):
    """
    Compute a smooth gradient using soft-max weighting of all facets.
    Vectorized implementation.
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    
    # Soft-max weights
    scaled_radii = radii / temperature
    scaled_radii -= np.max(scaled_radii)
    exp_weights = np.exp(scaled_radii)
    weights = exp_weights / (np.sum(exp_weights) + 1e-30)
    weights *= radii
    
    # Vectorized: for each facet, compute tangent direction for each of its 3 vertices
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    gradient = np.zeros_like(points)
    
    # For each vertex position, compute contribution
    for vi, pv in enumerate([p1, p2, p3]):
        # v = center - point, projected to tangent plane
        v = centers - pv
        dot_vp = np.einsum('ij,ij->i', v, pv)
        tangent_v = v - dot_vp[:, np.newaxis] * pv
        norm_tv = np.linalg.norm(tangent_v, axis=1)
        safe_norm = np.where(norm_tv > 1e-14, norm_tv, 1.0)
        direction = tangent_v / safe_norm[:, np.newaxis]
        # Only count where valid
        mask = norm_tv > 1e-14
        direction[~mask] = 0.0
        
        # Scatter-add weighted directions to the correct vertices
        idx = facets[:, vi]
        w = weights * mask
        np.add.at(gradient, idx, w[:, np.newaxis] * direction)
    
    return gradient

def _per_point_max_gradient(points, hull):
    """
    For each point, find the adjacent facet with maximum circumradius,
    and move the point toward that facet's circumcenter.
    Vectorized implementation.
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    
    point_max_radius = np.zeros(N)
    point_max_center = np.zeros((N, 3))
    
    # Vectorized max over facets per vertex
    for vi in range(3):
        idx = facets[:, vi]
        r = radii
        c = centers
        # For each vertex, find max radius among its facets
        # Use np.maximum.at for max reduction
        np.maximum.at(point_max_radius, idx, r)
    
    # Second pass to get the center corresponding to max radius
    # We need to track which facet gives the max for each point
    # Use a different approach: for each vertex, gather all its facet radii and centers
    point_max_center2 = np.zeros((N, 3))
    for vi in range(3):
        idx = facets[:, vi]
        # For each unique index, we want the center from the facet with max radius
        # This is harder to vectorize perfectly, so do a simple loop over facets
        pass
    
    # Simpler: iterate facets but vectorize the inner part
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
    
    # Compute gradient vectorized
    v = point_max_center - points
    dot_vp = np.einsum('ij,ij->i', v, points)
    tangent_v = v - dot_vp[:, np.newaxis] * points
    norm_tv = np.linalg.norm(tangent_v, axis=1)
    safe_norm = np.where(norm_tv > 1e-14, norm_tv, 1.0)
    gradient = tangent_v / safe_norm[:, np.newaxis] * point_max_radius[:, np.newaxis]
    mask = (norm_tv > 1e-14) & (point_max_radius > 0)
    gradient[~mask] = 0.0
    
    return gradient

def _compute_circumradii_fast(points, hull):
    """
    Fast vectorized circumradius computation for scoring.
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    return radii

def _normalize_points(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms

def _soft_max_objective_and_grad(pts_flat, temperature=0.01):
    """
    Objective and gradient for L-BFGS-B.
    pts_flat: (N*3,) flat array of 3D coordinates.
    Returns (objective, gradient) where objective is the soft-max covering radius.
    """
    pts = pts_flat.reshape(N, 3)
    pts = _normalize_points(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, np.zeros_like(pts_flat)
    
    radii, centers, facets = _compute_circumradii(pts, hull)
    
    # Soft-max objective: smooth approximation of max(radii)
    scaled = radii / temperature
    scaled -= np.max(scaled)
    exp_w = np.exp(scaled)
    soft_max_val = temperature * (np.log(np.sum(exp_w)) + np.max(scaled))
    
    # Gradient of soft-max w.r.t. radii
    weights = exp_w / (np.sum(exp_w) + 1e-30)
    
    # Gradient of radii w.r.t. points via the facet contributions
    # For each facet, the circumradius gradient w.r.t. each vertex
    # dr_k/dp_i for i in facet k
    # The circumcenter c_k moves as points move; dr = (c - p) . dc/dp + ... 
    # Simplified: use the same tangent direction logic
    grad_pts = np.zeros_like(pts)
    
    for k in range(len(facets)):
        w = weights[k]
        if w < 1e-10:
            continue
        c = centers[k]
        tri = facets[k]
        for idx in tri:
            p = pts[idx]
            v = c - p
            dot_vp = np.dot(v, p)
            tangent_v = v - dot_vp * p
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-14:
                grad_pts[idx] += w * tangent_v / norm_tv
    
    # Convert from point-space gradient to flat-space gradient
    # The normalization introduces a Jacobian; for small steps on sphere, 
    # the tangent gradient is approximately the flat gradient
    grad_flat = grad_pts.reshape(-1)
    
    return soft_max_val, grad_flat

def construct():
    from scipy.optimize import minimize
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize_points(points)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points = _normalize_points(points)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Coarse optimization with per-point max gradient
    step_size = 0.05
    min_step_size = 1e-7
    no_improve_count = 0
    max_no_improve = 40
    phase1_deadline = start_time + 50.0
    
    while time.time() < phase1_deadline:
        try:
            hull = ConvexHull(points)
        except Exception:
            kick = rng.normal(0, 0.02, points.shape)
            points += kick
            points = _normalize_points(points)
            continue
        
        grad = _per_point_max_gradient(points, hull)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points = _normalize_points(new_points)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            step_size = min(step_size * 1.03, 0.1)
        else:
            step_size = max(step_size * 0.85, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                kick = rng.normal(0, 0.03, points.shape)
                points += kick
                points = _normalize_points(points)
                step_size = 0.02
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    # Phase 2: L-BFGS-B optimization on soft-max surrogate
    temperature = 0.02
    phase2_deadline = start_time + 120.0
    
    # Use L-BFGS-B with the soft-max objective
    try:
        result = minimize(
            _soft_max_objective_and_grad,
            points.reshape(-1),
            args=(temperature,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_radius:
            new_pts = _normalize_points(result.x.reshape(N, 3))
            current_radius = _compute_covering_radius(new_pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_pts.copy()
    except Exception:
        pass
    
    # Check if we have time for another round
    if time.time() < phase2_deadline:
        # Try with lower temperature for refinement
        temperature = 0.005
        try:
            start_flat = best_points.reshape(-1)
            result = minimize(
                _soft_max_objective_and_grad,
                start_flat,
                args=(temperature,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_radius:
                new_pts = _normalize_points(result.x.reshape(N, 3))
                current_radius = _compute_covering_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = new_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with manual gradient steps
    if time.time() < deadline:
        step_size = 0.001
        temperature = 0.005
        no_improve_count = 0
        max_no_improve = 50
        points = best_points.copy()
        
        while time.time() < deadline:
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            grad = _soft_max_gradient(points, hull, temperature)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            new_points = points + step_size * grad
            new_points = _normalize_points(new_points)
            
            current_radius = _compute_covering_radius(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
            else:
                step_size = max(step_size * 0.95, 1e-8)
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
    
    return best_points
