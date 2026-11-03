"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    points = np.array(points)
    n = len(points)
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback: worst case
    
    # For each facet, find the circumradius (angular distance from facet center to vertices)
    # The covering radius is the maximum over all facets of their circumradius
    max_radius = 0.0
    
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # Find the point on the sphere that is equidistant from all facet vertices
        # and farthest from the origin (this is the circumcenter of the spherical triangle)
        
        # For a spherical triangle, the circumradius can be found by:
        # The circumcenter is the intersection of perpendicular bisectors
        # For a triangle with vertices a, b, c on unit sphere:
        # circumcenter direction = (a+b+c) normalized, but this only works for acute triangles
        
        # More robust: use the formula for spherical circumradius
        # For a spherical triangle, the circumradius R satisfies:
        # sin(R) = sin(a)sin(b)sin(c) / (2*sin(s)sin(s-a)sin(s-b)sin(s-c))
        # where a,b,c are side lengths and s is semiperimeter
        
        # Simpler approach: the circumradius of a facet is the angular distance
        # from the facet's circumcenter to any vertex
        # The circumcenter of a planar triangle is:
        # For points p1, p2, p3, circumcenter = solve for x such that |x-p1|=|x-p2|=|x-p3|
        
        p1, p2, p3 = facet_points[0], facet_points[1], facet_points[2]
        
        # Compute side lengths (angular distances)
        a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        
        s = (a + b + c) / 2.0
        
        # Spherical circumradius formula
        # sin(R) = sin(a)*sin(b)*sin(c) / (2*sin(s)*sin(s-a)*sin(s-b)*sin(s-c))
        sin_s = np.sin(s)
        sin_sa = np.sin(s - a)
        sin_sb = np.sin(s - b)
        sin_sc = np.sin(s - c)
        
        denom = 2.0 * sin_s * sin_sa * sin_sb * sin_sc
        if abs(denom) < 1e-15:
            # Degenerate case
            R = max(a, b, c) / 2.0
        else:
            sin_R = (np.sin(a) * np.sin(b) * np.sin(c)) / denom
            sin_R = np.clip(sin_R, 0, 1)
            R = np.arcsin(sin_R)
        
        max_radius = max(max_radius, R)
    
    return max_radius


def _smooth_covering_radius(points, temperature=0.05):
    """Smooth approximation of covering radius using log-sum-exp."""
    points = np.array(points)
    n = len(points)
    
    # For each pair, compute the angular distance
    # The covering radius is related to the maximum over all points on sphere of min distance to any chosen point
    # A proxy: for each chosen point, the "influence region" size
    
    # Better: compute for each facet its circumradius, then use log-sum-exp
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    radii = []
    for simplex in hull.simplices:
        facet_points = points[simplex]
        p1, p2, p3 = facet_points[0], facet_points[1], facet_points[2]
        
        a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        
        s = (a + b + c) / 2.0
        
        sin_s = np.sin(s)
        sin_sa = np.sin(s - a)
        sin_sb = np.sin(s - b)
        sin_sc = np.sin(s - c)
        
        denom = 2.0 * sin_s * sin_sa * sin_sb * sin_sc
        if abs(denom) < 1e-15:
            R = max(a, b, c) / 2.0
        else:
            sin_R = (np.sin(a) * np.sin(b) * np.sin(c)) / denom
            sin_R = np.clip(sin_R, 0, 1)
            R = np.arcsin(sin_R)
        
        radii.append(R)
    
    radii = np.array(radii)
    if len(radii) == 0:
        return np.pi
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    if max_r < 1e-10:
        return max_r
    
    log_sum_exp = max_r + np.log(np.sum(np.exp((radii - max_r) / temperature)))
    smooth_max = temperature * log_sum_exp
    
    return smooth_max


def _gradient_smooth_covering_radius(points, temperature=0.05):
    """Compute gradient of smooth covering radius w.r.t. points."""
    points = np.array(points, dtype=float)
    n = len(points)
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.zeros_like(points)
    
    radii = []
    facet_indices = []
    for simplex in hull.simplices:
        facet_points = points[simplex]
        p1, p2, p3 = facet_points[0], facet_points[1], facet_points[2]
        
        a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        
        s = (a + b + c) / 2.0
        
        sin_s = np.sin(s)
        sin_sa = np.sin(s - a)
        sin_sb = np.sin(s - b)
        sin_sc = np.sin(s - c)
        
        denom = 2.0 * sin_s * sin_sa * sin_sb * sin_sc
        if abs(denom) < 1e-15:
            R = max(a, b, c) / 2.0
        else:
            sin_R = (np.sin(a) * np.sin(b) * np.sin(c)) / denom
            sin_R = np.clip(sin_R, 0, 1)
            R = np.arcsin(sin_R)
        
        radii.append(R)
        facet_indices.append(simplex)
    
    radii = np.array(radii)
    if len(radii) == 0:
        return np.zeros_like(points)
    
    # Weights for log-sum-exp
    max_r = np.max(radii)
    if max_r < 1e-10:
        return np.zeros_like(points)
    
    exp_vals = np.exp((radii - max_r) / temperature)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient of R w.r.t. each point
    # R = arcsin(sin(a)*sin(b)*sin(c) / (2*sin(s)*sin(s-a)*sin(s-b)*sin(s-c)))
    # This is complex; use numerical gradient for the smooth part
    
    grad = np.zeros_like(points)
    
    for i, (R, simplex) in enumerate(zip(radii, facet_indices)):
        w = weights[i]
        if w < 1e-10:
            continue
        
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        
        s = (a + b + c) / 2.0
        
        sin_s = np.sin(s)
        sin_sa = np.sin(s - a)
        sin_sb = np.sin(s - b)
        sin_sc = np.sin(s - c)
        
        denom = 2.0 * sin_s * sin_sa * sin_sb * sin_sc
        if abs(denom) < 1e-15:
            continue
        
        # Numerical gradient for this facet's contribution
        eps = 1e-5
        for vi, idx in enumerate(simplex):
            orig = points[idx].copy()
            
            # Perturb in tangent plane
            for d in range(3):
                perturbed = orig.copy()
                perturbed[d] += eps
                # Project back to sphere
                perturbed = perturbed / np.linalg.norm(perturbed)
                
                points[idx] = perturbed
                
                # Recompute R for this facet
                p1_new, p2_new, p3_new = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                a_new = np.arccos(np.clip(np.dot(p2_new, p3_new), -1, 1))
                b_new = np.arccos(np.clip(np.dot(p1_new, p3_new), -1, 1))
                c_new = np.arccos(np.clip(np.dot(p1_new, p2_new), -1, 1))
                s_new = (a_new + b_new + c_new) / 2.0
                sin_s_new = np.sin(s_new)
                sin_sa_new = np.sin(s_new - a_new)
                sin_sb_new = np.sin(s_new - b_new)
                sin_sc_new = np.sin(s_new - c_new)
                denom_new = 2.0 * sin_s_new * sin_sa_new * sin_sb_new * sin_sc_new
                if abs(denom_new) < 1e-15:
                    R_new = max(a_new, b_new, c_new) / 2.0
                else:
                    sin_R_new = (np.sin(a_new) * np.sin(b_new) * np.sin(c_new)) / denom_new
                    sin_R_new = np.clip(sin_R_new, 0, 1)
                    R_new = np.arcsin(sin_R_new)
                
                dR_dpt = (R_new - R) / eps
                grad[idx][d] += w * temperature * dR_dpt
                
            points[idx] = orig
    
    # Project gradient to tangent plane
    for i in range(n):
        p = points[i]
        grad[i] -= np.dot(grad[i], p) * p
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Gradient-based optimization with smooth objective
    temperature = 0.05
    lr = 0.01
    n_iterations = 200
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Compute smooth covering radius and gradient
        smooth_r = _smooth_covering_radius(points, temperature)
        grad = _gradient_smooth_covering_radius(points, temperature)
        
        # Update points (gradient descent on covering radius)
        points = points - lr * grad
        
        # Project back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        points = points / norms
        
        # Compute actual covering radius
        actual_r = _covering_radius(points)
        
        if actual_r < best_radius:
            best_radius = actual_r
            best_points = points.copy()
        
        # Adaptive learning rate
        if it % 20 == 0 and it > 0:
            lr *= 0.9
    
    # Final: return best found
    return best_points
# EVOLVE-BLOCK-END
