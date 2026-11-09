"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _spherical_circumcenter_and_radius(a, b, c):
    """Compute circumcenter and circumradius of spherical triangle (a,b,c) on unit sphere."""
    # Circumcenter x satisfies x·a = x·b = x·c, |x|=1
    # x is parallel to (a-b) x (b-c)
    v1 = a - b
    v2 = b - c
    x = np.cross(v1, v2)
    norm_x = np.linalg.norm(x)
    if norm_x < 1e-12:
        # Degenerate - return midpoint direction
        x = a + b + c
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-12:
            return a, 0.0
    x = x / norm_x
    # Check if x is on the correct side (same hemisphere as the triangle)
    # The circumradius is the angle from x to a
    cos_r = np.clip(np.dot(x, a), -1.0, 1.0)
    r = np.arccos(cos_r)
    # Make sure r is the smaller angle (circumradius should be < pi/2 for small caps)
    if r > np.pi / 2:
        x = -x
        r = np.pi - r
    return x, r

def _compute_facet_circumradii(points):
    """Compute circumradii for all facets of the convex hull."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None
    
    radii = []
    centers = []
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        center, r = _spherical_circumcenter_and_radius(a, b, c)
        radii.append(r)
        centers.append(center)
    
    return np.array(radii), np.array(centers)

def _soft_max(values, beta=10.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def _objective_and_grad(points, beta=50.0):
    """Compute soft-max of facet circumradii and gradient."""
    radii, centers = _compute_facet_circumradii(points)
    if radii is None:
        return 1e6, np.zeros_like(points)
    
    # Soft max of radii
    m = np.max(radii)
    exp_vals = np.exp(beta * (radii - m))
    sum_exp = np.sum(exp_vals)
    obj = m + np.log(sum_exp) / beta
    
    # Gradient: d(obj)/d(point_i)
    # obj = m + (1/beta) * log(sum_j exp(beta * r_j))
    # d(obj)/d(r_j) = exp(beta * r_j) / sum_exp
    # d(r_j)/d(point_i) for facet j containing point i
    
    grad = np.zeros_like(points)
    weights = exp_vals / sum_exp  # d(obj)/d(r_j)
    
    for idx, simplex in enumerate(ConvexHull(points).simplices):
        w = weights[idx]
        if w < 1e-15:
            continue
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        center, r = _spherical_circumcenter_and_radius(a, b, c)
        
        # dr/d(point) for each of the 3 points
        # r = arccos(center · a), so dr/da = -center / sin(r) (projected)
        # But we need the gradient on the sphere (tangent space)
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        # Gradient of r w.r.t. point a: -center/sin(r) projected to tangent space at a
        # tangent projection: v - (v·a)a
        for p_idx, p in zip(simplex, [a, b, c]):
            g_r = -center / sin_r
            # Project to tangent space at p
            g_tangent = g_r - np.dot(g_r, p) * p
            grad[p_idx] += w * g_tangent
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_score = 1e6
    
    # Evaluate initial
    radii, _ = _compute_facet_circumradii(points)
    if radii is not None:
        best_score = np.max(radii) * 180.0 / np.pi
    
    # Gradient descent with adaptive step size
    lr = 0.01
    beta = 20.0  # Start with moderate sharpness
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Increase beta over time for sharper max
        beta = min(50.0, 20.0 + iteration * 0.01)
        
        obj, grad = _objective_and_grad(points, beta)
        
        if not np.all(np.isfinite(grad)):
            lr *= 0.5
            continue
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Normalize gradient for stable updates
        grad = grad / (grad_norm + 1e-12)
        
        # Update points
        new_points = points - lr * grad
        
        # Project back to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check if improvement
        new_radii, _ = _compute_facet_circumradii(new_points)
        if new_radii is None:
            lr *= 0.5
            continue
        
        new_max = np.max(new_radii)
        
        if new_max < best_score * np.pi / 180.0:
            best_score = new_max * 180.0 / np.pi
            best_points = new_points.copy()
            points = new_points
            lr = min(lr * 1.05, 0.05)  # Slightly increase step
        else:
            lr *= 0.7  # Reduce step
        
        # Early termination if converged
        if best_score < 18.31 and iteration > 1000:
            break
    
    # Final polish: local refinement with smaller steps
    if time.time() < deadline - 10.0:
        points = best_points.copy()
        lr = 0.001
        for _ in range(2000):
            if time.time() > deadline:
                break
            obj, grad = _objective_and_grad(points, beta=100.0)
            if not np.all(np.isfinite(grad)):
                break
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad = grad / grad_norm
            new_points = points - lr * grad
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            
            new_radii, _ = _compute_facet_circumradii(new_points)
            if new_radii is None:
                lr *= 0.5
                continue
            new_max = np.max(new_radii)
            if new_max < best_score * np.pi / 180.0:
                best_score = new_max * 180.0 / np.pi
                best_points = new_points.copy()
                points = new_points
            else:
                lr *= 0.8
    
    return best_points
# EVOLVE-BLOCK-END
