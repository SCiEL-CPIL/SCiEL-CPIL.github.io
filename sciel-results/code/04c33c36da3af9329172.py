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
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = np.where(dist > np.pi / 2, np.pi - dist, dist)
    
    return radii, centers, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        radii, centers, facets = _compute_circumradii(points, hull)
        return float(np.max(radii)) * 180.0 / np.pi
    except Exception:
        return 180.0

def _soft_max_objective_and_grad(points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    
    Returns:
        obj: scalar soft-max objective
        grad: (N, 3) gradient
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
    radii, centers, facets = _compute_circumradii(points, hull)
    M = len(radii)
    
    if M == 0:
        return 0.0, np.zeros_like(points)
    
    # Log-sum-exp for numerical stability
    r_max = np.max(radii)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    # Soft max: (1/beta) * log(sum(exp(beta * r_i)))
    # Use shifted version for stability
    shifted = beta * (radii - r_max)
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    # Gradient of soft-max with respect to each radius
    # d/d r_i [ (1/beta) log(sum_j exp(beta r_j)) ] = exp(beta r_i) / sum_j exp(beta r_j)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Now compute gradient with respect to point positions
    # For each facet, the circumradius depends on the three vertices
    # The gradient of the circumradius with respect to a vertex can be computed
    # via the chain rule through the circumcenter
    
    grad = np.zeros_like(points)
    
    for k in range(M):
        if weights[k] < 1e-10:
            continue
        
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        c = centers[k]
        r_k = radii[k]
        
        if r_k < 1e-12:
            continue
        
        # The circumradius r = arccos(dot(c, p1))
        # d r / d p1 = -c / sqrt(1 - dot(c,p1)^2) (projected to tangent plane)
        # But we need to be careful about the normalization constraint
        
        # For a point p on the unit sphere, the gradient of arccos(dot(c, p))
        # with respect to p (in R^3) is:
        # -c / sin(arccos(dot(c,p))) * (I - p p^T) [projection to tangent]
        
        dot_cp1 = np.dot(c, p1)
        sin_r1 = np.sqrt(max(0.0, 1.0 - dot_cp1 * dot_cp1))
        
        if sin_r1 > 1e-12:
            # Gradient of r with respect to p1 in R^3
            grad_p1 = -c / sin_r1
            # Project to tangent plane at p1
            grad_p1 -= np.dot(grad_p1, p1) * p1
            grad[tri[0]] += weights[k] * grad_p1
        
        # Similarly for p2 and p3
        dot_cp2 = np.dot(c, p2)
        sin_r2 = np.sqrt(max(0.0, 1.0 - dot_cp2 * dot_cp2))
        if sin_r2 > 1e-12:
            grad_p2 = -c / sin_r2
            grad_p2 -= np.dot(grad_p2, p2) * p2
            grad[tri[1]] += weights[k] * grad_p2
        
        dot_cp3 = np.dot(c, p3)
        sin_r3 = np.sqrt(max(0.0, 1.0 - dot_cp3 * dot_cp3))
        if sin_r3 > 1e-12:
            grad_p3 = -c / sin_r3
            grad_p3 -= np.dot(grad_p3, p3) * p3
            grad[tri[2]] += weights[k] * grad_p3
    
    return float(log_sum_exp), grad

def _normalize_points(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def construct():
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
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Gradient descent with soft-max surrogate
    step_size = 0.02
    min_step_size = 1e-8
    beta = 20.0  # Start with moderate sharpness
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline - 30.0:  # Leave 30s for polish
        
        obj, grad = _soft_max_objective_and_grad(points, beta)
        
        # Project gradient to tangent space and normalize
        grad_tangent = np.zeros_like(points)
        for i in range(N):
            g = grad[i]
            g_tan = g - np.dot(g, points[i]) * points[i]
            norm_g = np.linalg.norm(g_tan)
            if norm_g > 1e-12:
                grad_tangent[i] = g_tan / norm_g
        
        # Gradient descent step
        new_points = points - step_size * grad_tangent
        new_points = _normalize_points(new_points)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Increase step size slightly if making progress
            step_size = min(step_size * 1.03, 0.05)
            # Increase beta to sharpen the surrogate
            beta = min(beta * 1.05, 200.0)
        else:
            step_size = max(step_size * 0.85, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Stagnation: random kick
                kick = rng.normal(0, 0.03, points.shape)
                points += kick
                points = _normalize_points(points)
                step_size = 0.01
                beta = max(beta / 1.5, 10.0)
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    # Phase 2: High-precision polish using exact max and Nelder-Mead style
    points = best_points.copy()
    polish_deadline = deadline - 5.0
    polish_step = 0.005
    
    while time.time() < polish_deadline:
        try:
            hull = ConvexHull(points)
            radii, centers, facets = _compute_circumradii(points, hull)
        except Exception:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        worst_facet = facets[max_idx]
        worst_center = centers[max_idx]
        
        # Move points in the worst facet towards the circumcenter
        for idx in worst_facet:
            v = worst_center - points[idx]
            dot_vp = np.dot(v, points[idx])
            tangent_v = v - dot_vp * points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                points[idx] += polish_step * tangent_v / norm_tv
        
        points = _normalize_points(points)
        
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # Revert and try smaller step
            polish_step *= 0.7
            if polish_step < 1e-6:
                break
            # Restore points
            try:
                hull = ConvexHull(best_points)
                radii, centers, facets = _compute_circumradii(best_points, hull)
                max_idx = np.argmax(radii)
                worst_facet = facets[max_idx]
                worst_center = centers[max_idx]
                points = best_points.copy()
                for idx in worst_facet:
                    v = worst_center - points[idx]
                    dot_vp = np.dot(v, points[idx])
                    tangent_v = v - dot_vp * points[idx]
                    norm_tv = np.linalg.norm(tangent_v)
                    if norm_tv > 1e-12:
                        points[idx] += polish_step * tangent_v / norm_tv
                points = _normalize_points(points)
            except Exception:
                break
    
    return best_points
