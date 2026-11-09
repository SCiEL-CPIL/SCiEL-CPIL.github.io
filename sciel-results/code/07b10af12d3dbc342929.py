import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_face_data(pts):
    """
    Compute circumradius, center, and vertex indices for all faces of the convex hull.
    Returns:
    radii: array of radii
    centers: array of centers
    simplices: list of vertex indices for each face
    """
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return None, None, None
        
    simplices = hull.simplices
    n_faces = len(simplices)
    radii = np.zeros(n_faces)
    centers = np.zeros((n_faces, 3))
    
    # Vectorized circumcenter calculation
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    v1 = a - b
    v2 = b - c
    cross = np.cross(v1, v2)
    norms = np.linalg.norm(cross, axis=1)
    
    # Handle degenerate cases (norms close to 0)
    valid = norms > 1e-10
    
    if np.any(valid):
        centers_valid = cross[valid] / norms[valid, np.newaxis]
        centroids_valid = (a[valid] + b[valid] + c[valid]) / 3.0
        dots = np.sum(centers_valid * centroids_valid, axis=1)
        # Flip sign if dot product is negative
        flip = dots < 0
        centers_valid[flip] *= -1
        
        cos_r = np.sum(centers_valid * a[valid], axis=1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
        centers[valid] = centers_valid
        
    if np.any(~valid):
        # Degenerate: use centroid
        centers_invalid = (a[~valid] + b[~valid] + c[~valid]) / 3.0
        norms_c = np.linalg.norm(centers_invalid, axis=1)
        valid_c = norms_c > 1e-10
        centers_invalid[valid_c] /= norms_c[valid_c, np.newaxis]
        centers[~valid] = centers_invalid
        
        cos_r = np.sum(centers_invalid[valid_c] * a[~valid][valid_c], axis=1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        radii[~valid][valid_c] = np.arccos(cos_r)
        radii[~valid][~valid_c] = np.pi

    return radii, centers, simplices

def _compute_covering_radius(pts):
    radii, _, _ = _compute_face_data(pts)
    if radii is None:
        return np.pi
    return np.max(radii)

def _riemannian_steepest_descent(pts, step_size, beta=10.0):
    """
    Riemannian steepest descent on the sphere for the covering radius.
    """
    radii, centers, simplices = _compute_face_data(pts)
    if radii is None:
        return pts
        
    # Soft-max weights
    max_r = np.max(radii)
    weights = np.exp(beta * (radii - max_r))
    weights /= np.sum(weights)
    
    # Build gradient in ambient space
    grad = np.zeros_like(pts)
    
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-10:
            continue
            
        center = centers[k]
        tri = simplices[k]
        
        for idx in tri:
            v = pts[idx]
            # Direction from vertex to circumcenter
            diff = center - v
            # Project onto tangent space at v
            dot_v = np.dot(diff, v)
            tangent = diff - dot_v * v
            grad[idx] += w * tangent
            
    # Normalize and apply step on the sphere
    # Use global normalization for stability
    max_grad_norm = np.max(np.linalg.norm(grad, axis=1))
    if max_grad_norm > 1e-12:
        grad_normalized = grad / max_grad_norm
        new_pts = pts + step_size * grad_normalized
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
    else:
        new_pts = pts.copy()
        
    return new_pts

def _local_refine_riemannian(pts, deadline, max_iters=2000, initial_step=0.05, beta=10.0):
    """
    Local refinement using Riemannian steepest descent on the sphere.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        new_pts = _riemannian_steepest_descent(pts, step_size, beta)
        
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-12:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            # Decay step size slowly
            step_size *= 0.999
        else:
            no_improve_count += 1
            step_size *= 0.95
            if step_size < 1e-6 or no_improve_count > 50:
                break
                
    return best_pts, best_radius

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. We use 145s for computation.
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Initial points: Fibonacci sphere
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Phase 1: Riemannian refinement
    if time.time() < deadline - 20:
        refined_pts, refined_radius = _local_refine_riemannian(points.copy(), deadline - 20, max_iters=3000, initial_step=0.05, beta=10.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    # Phase 2: Multiple restarts with different initializations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            pts = best_points + np.random.randn(N, 3) * 0.05
        elif restart == 1:
            pts = best_points + np.random.randn(N, 3) * 0.1
        elif restart == 2:
            # Random
            pts = np.random.randn(N, 3)
        elif restart == 3:
            # Fibonacci with phase shift
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 4
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Perturbed best
            pts = best_points + np.random.randn(N, 3) * (0.05 + 0.02 * restart)
            
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Refine
        if time.time() < deadline - 5:
            refined_pts, refined_radius = _local_refine_riemannian(pts, deadline - 5, max_iters=1000, initial_step=0.05, beta=10.0)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
                
    # Final polish with higher beta and smaller step
    if time.time() < deadline - 2:
        refined_pts, refined_radius = _local_refine_riemannian(best_points.copy(), deadline - 2, max_iters=500, initial_step=0.01, beta=20.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
